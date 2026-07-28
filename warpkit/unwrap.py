import logging
from types import SimpleNamespace
from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    generate_binary_structure,
)
from scipy.stats import mode
from skimage.filters import threshold_otsu  # type: ignore

from .concurrency import run_executor
from .model import weighted_regression
from .utilities import (
    corr2_coeff,
    create_brain_mask,
    get_largest_connected_component,
    rescale_phase,
)
from .warpkit_cpp import romeo_unwrap3d, romeo_unwrap4d, romeo_voxelquality

# Candidate global 2*pi branches scanned by the intercept selector. One wrap is
# 1/dTE Hz of global field (~59 Hz for a 17 ms echo spacing), and ROMEO's
# ``correct_global`` already pins |median field| below 1/(2*dTE), so +/-1 covers
# everything reachable in practice.
BRANCH_CANDIDATES = (-1, 0, 1)


def reject_outliers(data, m=2.0):
    """Reject outliers from data."""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    return data[s < m]


def get_dual_echo_fieldmap(phases, tes, mags, mask):
    """Compute the fieldmap from the first two echoes.

    Parameters
    ----------
    phases : np.ndarray
        Array of unwrapped phase data with shape (x, y, z, echo)
    tes : np.ndarray
        Echo times associated with each phase
    mags : np.ndarray
        Array of magnitude data with shape (x, y, z, echo)
    mask : np.ndarray
        Mask of voxels to use for unwrapping

    Returns
    -------
    np.ndarray of shape (x, y, z)
        Fieldmap in Hz
    np.ndarray of shape (x, y, z, echo)
        Unwrapped phases
    """
    # unwrap the phases
    unwrapped_phases = romeo_unwrap4d(
        phase=phases,
        tes=tes,
        weights="romeo",
        mag=mags,
        mask=mask,
        correct_global=True,
        maxseeds=1,
        merge_regions=False,
        correct_regions=False,
    )

    phase_diff = unwrapped_phases[..., 1] - unwrapped_phases[..., 0]
    fieldmap = (1000 / (2 * np.pi)) * phase_diff / (tes[1] - tes[0])
    return fieldmap, unwrapped_phases


def _evaluate_branch(
    n_wraps: int,
    unwrapped_diff: npt.NDArray[np.float32],
    phase0: npt.NDArray[np.float32],
    phase1: npt.NDArray[np.float32],
    mag0: npt.NDArray[np.float32],
    mag1: npt.NDArray[np.float32],
    te0: np.float32 | float,
    te1: np.float32 | float,
    mask: npt.NDArray[np.bool_],
    score_mask: npt.NDArray[np.bool_],
):
    """Reconstruct one candidate global 2*pi branch and score how well it fits.

    Shifting the unwrapped phase difference by ``2*pi*n_wraps`` moves the
    MCPC-3D-S phase offset by ``wrap(2*pi*n_wraps*te0/dTE)``. That lands as the
    same additive constant on every echo, so the echoes no longer extrapolate
    back through zero at TE=0. Fitting a line through the two echoes and
    reading off its intercept measures that constant directly: the right branch
    gives ~0, a wrong branch gives roughly one ``_branch_intercept_step``.

    The intercept is taken as a median over ``score_mask`` because the error is
    a single global constant, not a per-voxel effect.

    Returns
    -------
    intercept : float
        |median intercept| in radians over ``score_mask``. Zero means the
        echoes are proportional to TE, which is what a correct offset gives.
    offset : npt.NDArray[np.float32]
        Phase offset implied by this branch.
    fieldmap : npt.NDArray[np.float32]
        Dual-echo field map in Hz implied by this branch.
    unwrapped_phases : npt.NDArray[np.float32]
        Dual-echo unwrapped phases implied by this branch.
    """
    shifted = unwrapped_diff + 2 * np.pi * n_wraps
    offset = np.angle(np.exp(1j * (phase0 - ((te0 * shifted) / (te1 - te0)))))
    proposed_phases = (
        np.stack([phase0, phase1], axis=-1) - offset[..., np.newaxis]
    ).astype(np.float32)
    fieldmap, unwrapped_phases = get_dual_echo_fieldmap(
        proposed_phases,
        np.array([te0, te1], dtype=np.float32),
        np.stack([mag0, mag1], axis=-1).astype(np.float32),
        mask,
    )
    y0 = unwrapped_phases[score_mask, 0].astype(np.float64)
    y1 = unwrapped_phases[score_mask, 1].astype(np.float64)
    if y0.size == 0:
        return float("inf"), offset, fieldmap, unwrapped_phases
    slope = (y1 - y0) / (float(te1) - float(te0))
    intercept = float(abs(np.median(y0 - slope * float(te0))))
    return intercept, offset, fieldmap, unwrapped_phases


def _weighted_median(values: npt.NDArray, weights: npt.NDArray) -> float:
    """Weighted median of ``values``. Used with magnitude-squared weights so
    low-SNR voxels do not sway the global field estimate.

    A median, not a mean or an M-estimator. The field distribution over the
    brain is wide (~2 wraps) and right-skewed, and the 1/(2*dTE) threshold this
    feeds sits *inside* its bulk, between the 50th and 75th percentiles. So the
    question the threshold asks -- is most of the brain within half a wrap of
    zero -- is a question about the 50% point, and the median is the statistic
    that answers it. Anything pulled toward the mean reads ~2 Hz higher and
    moves the estimate toward the boundary for no gain in accuracy: a Huber
    M-estimator tried here cut the worst-case margin from 1.5 Hz to 0.05 Hz.

    Nor an argmax. The threshold comparison has to give the same answer on
    consecutive frames, and a median moves smoothly with the data; field
    distributions are often bimodal and a near-tie between peaks makes an
    argmax flip frame to frame. A half-sample mode was tried and reverted for
    exactly that reason.
    """
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _branch_intercept_step(te0: np.float32 | float, te1: np.float32 | float) -> float:
    """Intercept, in radians, that one wrap of branch error introduces.

    Candidate branches are spaced exactly this far apart in intercept, so it is
    the natural scale for "this branch does not fit". It depends only on the
    echo times -- no data required.

    Returns 0.0 when te0/dTE is an integer: there the branch does not move the
    phase offset at all, so the candidates are indistinguishable (and the choice
    cannot affect the output either).
    """
    dte = float(te1) - float(te0)
    if dte <= 0:
        return 0.0
    return float(abs(np.angle(np.exp(1j * 2 * np.pi * float(te0) / dte))))


def _select_branch(
    unwrapped_diff: npt.NDArray[np.float32],
    phase0: npt.NDArray[np.float32],
    phase1: npt.NDArray[np.float32],
    mag0: npt.NDArray[np.float32],
    mag1: npt.NDArray[np.float32],
    te0: np.float32 | float,
    te1: np.float32 | float,
    mask: npt.NDArray[np.bool_],
    score_mask: npt.NDArray[np.bool_],
) -> tuple[int, dict[int, float]]:
    """Pick the global 2*pi branch, in two stages.

    1. **Consistency.** A branch carrying a leftover intercept is not a valid
       explanation of the data at all, so discard those. If exactly one
       candidate survives it is the answer, whatever its field looks like. If
       none survives, return 0 -- a failed fit is not evidence for any branch.
    2. **Prior, only to break a tie.** Depending on TE0/dTE several branches can
       be exactly through-origin; the alias is genuinely reachable and no
       statistic computed from the phase can separate them. Among the survivors,
       take the smallest weighted-median field.

    Stage 2 is the same prior ``correct_global`` already applies, but on a much
    better conditioned statistic: weighted by ``mag1**2``, over an eroded brain
    mask, on the field itself rather than an unweighted median of rounded wrap
    counts over a dilated mask. That difference is the whole point.
    ``correct_global``'s ballot becomes unstable when the global field sits near
    1/(2*dTE), and then it flips frame to frame.

    The weight is the *second* echo's magnitude because the field comes from
    ``phase1 - phase0``, whose noise is dominated by the weaker echo. Weighting
    on ``mag0`` instead lets voxels with fast T2* decay -- a healthy echo 0 and a
    collapsed echo 1, i.e. exactly the air-tissue interfaces -- carry full
    weight. Measured over 29 runs, this does not move the decision boundary
    (that is fixed at 1/(2*dTE)) and barely moves the margin; what it buys is
    stability. The estimator's offset varies by 0.43 Hz across subjects under
    ``mag1**2`` against 0.99 Hz under ``mag0**2``, and that between-subject
    spread, not the per-subject margin, is what predicts a wrap flip.

    Measured on `ds006131` sub-20828 (k = 0.5742, wrap 40.44 Hz, half-wrap
    20.22 Hz), whose field sits at 17.7 Hz -- 2.5 Hz under the boundary. On 19
    of 243 frames ``correct_global``'s median jumps to +/-1 and the dual-echo
    field flips from +16.2 Hz to -22.8 Hz, a full wrap, for a single frame at a
    time. The intercept test detects that something moved (the fitting set goes
    from {-1, 0} to {0, +1}) but cannot rank the two survivors. The prior can:
    branch +1 restores +17.7 Hz against branch 0's -22.8 Hz, so it fixes all 19
    and leaves the other 224 untouched.

    The prior is consulted *only* between branches that already fit, so it can
    never select something the data contradicts.
    """
    evaluated = {
        n: _evaluate_branch(
            n, unwrapped_diff, phase0, phase1, mag0, mag1, te0, te1, mask, score_mask
        )
        for n in BRANCH_CANDIDATES
    }
    scores = {n: e[0] for n, e in evaluated.items()}
    # Calibrate "how big is a wrong branch" two ways and take the stricter.
    # The analytic scale depends only on the TEs, so it still calibrates when
    # every candidate happens to fit; the observed scale stays honest when
    # ROMEO absorbs part of the intercept into its own 2*pi steps and the real
    # penalty comes out smaller than theory predicts. Taking the minimum makes
    # the consistency test harder to pass, which biases toward doing nothing.
    step = _branch_intercept_step(te0, te1)
    if step <= 0:
        # te0/dTE is an integer: the branch does not move the phase offset, so
        # every candidate returns the same result and there is nothing to pick.
        return 0, scores
    scale = min(max(scores.values()), step)
    if not np.isfinite(scale) or scale <= 0:
        return 0, scores
    # Nearest-neighbour on the lattice: candidate intercepts sit at 0 or at one
    # ``scale``, so the boundary is the midpoint. Not a tuned threshold -- the
    # two clusters are separated by 6-9 orders of magnitude on measured data, so
    # any boundary strictly inside (0, scale) gives the same answer. The
    # midpoint is simply the one that needs no justifying, and it keeps a branch
    # that is most of a wrap out from ever counting as a fit.
    cutoff = scale / 2
    consistent = [n for n, s in scores.items() if s < cutoff]
    if not consistent:
        # nothing fits; a failed fit is not evidence for any branch
        return 0, scores
    if len(consistent) == 1:
        return consistent[0], scores
    # Tie: every survivor explains the phase equally well, so fall back to the
    # prior and take the one closest to zero global field.
    weights = np.square(mag1[score_mask].astype(np.float64))
    fields = {
        n: _weighted_median(evaluated[n][2][score_mask].astype(np.float64), weights)
        for n in consistent
    }
    return min(consistent, key=lambda n: abs(fields[n])), scores


def mcpc_3d_s(
    mag0: npt.NDArray[np.float32],
    mag1: npt.NDArray[np.float32],
    phase0: npt.NDArray[np.float32],
    phase1: npt.NDArray[np.float32],
    te0: np.float32 | float,
    te1: np.float32 | float,
    mask: npt.NDArray[np.bool_],
):
    """Apply the MCPC-3D-S algorithm to compute the phase offset.

    Parameters
    ----------
    mag0 : npt.NDArray[np.float32]
        Magnitude image for the first echo
    mag1 : npt.NDArray[np.float32]
        Magnitude image for the second echo
    phase0 : npt.NDArray[np.float32]
        Phase image for the first echo
    phase1 : npt.NDArray[np.float32]
        Phase image for the second echo
    te0 : np.float32 | float
        Echo time for the first echo
    te1 : np.float32 | float
        Echo time for the second echo
    mask : npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping

    Returns
    -------
    npt.NDArray[np.float32]
        Phase offset in radians
    npt.NDArray[np.float32]
        Unwrapped difference in phase
    """
    signal_diff = mag0 * mag1 * np.exp(1j * (phase1 - phase0))
    mag_diff = np.abs(signal_diff)
    phase_diff = np.angle(signal_diff)
    unwrapped_diff = romeo_unwrap3d(
        phase=phase_diff,
        weights="romeo",
        mag=mag_diff,
        mask=mask,
        correct_global=True,
    )
    voxel_mask = create_brain_mask(mag0, -2)

    n_wraps, scores = _select_branch(
        unwrapped_diff, phase0, phase1, mag0, mag1, te0, te1, mask, voxel_mask
    )
    logging.info(
        "branch selection: n=%+d (intercepts=%s)",
        n_wraps,
        {n: f"{s_:.4e}" for n, s_ in scores.items()},
    )

    unwrapped_diff = unwrapped_diff + 2 * np.pi * n_wraps

    # compute the phase offset
    return np.angle(
        np.exp(1j * (phase0 - ((te0 * unwrapped_diff) / (te1 - te0))))
    ), unwrapped_diff


def unwrap_phase(
    phase_data: npt.NDArray[np.float32],
    mag_data: npt.NDArray[np.float32],
    tes: npt.NDArray[np.float32],
    mask_data: npt.NDArray[np.bool_],
    automask: bool = True,
    automask_dilation: int = 3,
    idx: int | None = None,
    debug: bool = False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    """Unwraps the phase for a single frame of ME-EPI data.

    Parameters
    ----------
    phase_data : npt.NDArray[np.float32]
        Single frame of phase data with shape (x, y, z, echo)
    mag_data : npt.NDArray[np.float32]
        Single frame of magnitude data with shape (x, y, z, echo)
    tes : npt.NDArray[np.float32]
        Echo times associated with each phase
    mask_data : npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping
    automask : bool, optional
        Automatically compute a mask, by default True
    automask_dilation : int, optional
        Number of extra dilations (or erosions if negative) to perform, by default 3
    idx : int, optional
        Index of the frame being processed for verbosity, by default None

    Returns
    -------
    npt.NDArray[np.float32]
        unwrapped phase in radians
    npt.NDArray[np.int8]
        mask
    """
    if idx is not None:
        logging.info(f"Processing frame: {idx}")

    # if automask is True, generate a mask for the frame, instead of using mask_data
    if automask:
        # the theory goes like this, the magnitude/otsu base mask can be too aggressive occasionally
        # and the voxel quality mask can get extra voxels that are not brain, but is noisy
        # so we combine the two masks to get a better mask
        vq = romeo_voxelquality(
            phase_data, tes, np.ones(shape=mag_data.shape, dtype=np.float32)
        )

        vq_mask = vq > threshold_otsu(vq)
        strel = generate_binary_structure(3, 2)
        vq_mask = cast(npt.NDArray[np.bool_], binary_fill_holes(vq_mask, strel))
        # get largest connected component
        vq_mask = get_largest_connected_component(vq_mask)

        # combine masks
        echo_idx = np.argmin(tes)
        mag_shortest = mag_data[..., echo_idx]
        brain_mask = create_brain_mask(mag_shortest)
        combined_mask = brain_mask | vq_mask
        combined_mask = get_largest_connected_component(combined_mask)

        # erode then dilate
        combined_mask = cast(
            npt.NDArray[np.bool_], binary_erosion(combined_mask, strel, iterations=2)
        )
        combined_mask = get_largest_connected_component(combined_mask)
        combined_mask = cast(
            npt.NDArray[np.bool_], binary_dilation(combined_mask, strel, iterations=2)
        )

        # get a dilated verision of the mask
        combined_mask_dilated = cast(
            npt.NDArray[np.bool_],
            binary_dilation(combined_mask, strel, iterations=automask_dilation),
        )

        # get sum of masks (we can select dilated vs original version by indexing)
        mask_data_select = combined_mask.astype(np.int8) + combined_mask_dilated.astype(
            np.int8
        )

        # let mask_data be the dilated version
        mask_data = mask_data_select > 0

    # Do MCPC-3D-S algo to compute phase offset
    phase_offset, unwrapped_diff = mcpc_3d_s(
        mag_data[..., 0],
        mag_data[..., 1],
        phase_data[..., 0],
        phase_data[..., 1],
        tes[0],
        tes[1],
        mask_data,
    )
    if debug:
        global affine
        global header
        nib.Nifti1Image(phase_offset, affine, header).to_filename(
            f"phase_offset{idx}.nii"
        )
        nib.Nifti1Image(unwrapped_diff, affine, header).to_filename(f"ud{idx}.nii")

    # remove phase offset from data
    phase_data -= phase_offset[..., np.newaxis]

    # unwrap the phase data
    unwrapped = romeo_unwrap4d(
        phase=phase_data,
        tes=tes,
        weights="romeo",
        mag=mag_data,
        mask=mask_data,
        correct_global=True,
        maxseeds=1,
        merge_regions=False,
        correct_regions=False,
    )

    # global mode correction
    # this computes the global mode offset for the first echo then tries to find the offset
    # that minimizes the residuals for each subsequent echo
    # use auto mask to get brain mask
    echo_idx = np.argmin(tes)
    mag_shortest = mag_data[..., echo_idx]
    brain_mask = create_brain_mask(mag_shortest)

    # for each of these matrices tes are on rows, voxels are columns
    # get design matrix
    x_mat = tes[:, np.newaxis]

    # get magnitude weight matrix
    w_mat = mag_data[brain_mask, :].T

    # loop over each index past 1st echo
    for i_echo in range(1, tes.shape[0]):
        # get matrix with the masked unwrapped data (weighted by magnitude)
        y_mat = unwrapped[brain_mask, :].T

        # Compute offset through linear regression method
        best_offset = compute_offset(i_echo, w_mat, x_mat, y_mat)

        # apply the offset
        unwrapped[..., i_echo] += 2 * np.pi * best_offset

    # set anything outside of mask_data to 0
    unwrapped[~mask_data] = 0

    # set final mask to return
    final_mask = mask_data_select if automask else mask_data.astype(np.int8)  # type: ignore

    # return the unwrapped data
    return unwrapped, final_mask


def check_temporal_consistency_corr(
    unwrapped_data: npt.NDArray,
    unwrapped_echo_1: npt.NDArray,
    tes,
    mag: list[nib.Nifti1Image],
    t,
    frame_idx,
    masks: npt.NDArray,
    threshold: float = 0.98,
):
    """Ensures phase unwrapping solutions are temporally consistent

    This uses correlation as a similarity metric between frames to enforce temporal consistency.

    Parameters
    ----------
    unwrapped_data : npt.NDArray
        unwrapped phase data, where last column is time, and second to last column are the echoes.
        This array will be modified in place by this function.
    tes : npt.NDArray
        echo times
    mag : List[nib.Nifti1Image]
        magnitude images
    frames : List[int]
        list of frames that are being processed
    threshold : float
        threshold for correlation similarity. By default 0.98
    """
    logging.info(f"Computing temporal consistency check for frame: {t}")

    # generate brain mask (with 1 voxel erosion)
    echo_idx = np.argmin(tes)
    mag_shortest = mag[echo_idx].dataobj[..., frame_idx]
    brain_mask = create_brain_mask(mag_shortest, -1)

    # get the current frame phase
    current_frame_data = unwrapped_echo_1[brain_mask, t][:, np.newaxis]

    # get the correlation between the current frame and all other frames
    corr = corr2_coeff(current_frame_data, unwrapped_echo_1[brain_mask, :]).ravel()

    # threhold the RD
    tmask = corr > threshold

    # get indices of mask
    indices = np.where(tmask)[0]

    # get mask for frame
    mask = masks[..., t] > 0

    # for each frame compute the mean value along the time axis (masked by indices and mask)
    mean_voxels = np.mean(unwrapped_echo_1[mask][:, indices], axis=-1)

    # for this frame figure out the integer multiple that minimizes the value to the mean voxel
    int_map = np.round((mean_voxels - unwrapped_echo_1[mask, t]) / (2 * np.pi)).astype(
        int
    )

    # correct the data using the integer map
    unwrapped_data[mask, 0, t] += 2 * np.pi * int_map

    # format weight matrix
    weights_mat = np.stack([m.dataobj[..., frame_idx] for m in mag], axis=-1)[mask].T

    # form design matrix
    x_mat = tes[:, np.newaxis]

    # fit subsequent echos to the weighted linear regression from the first echo
    for echo in range(1, unwrapped_data.shape[-2]):
        # form response matrix
        y_mat = unwrapped_data[mask, :echo, t].T

        # fit model to data
        coefficients, _ = weighted_regression(x_mat[:echo], y_mat, weights_mat[:echo])

        # get the predicted values for this echo
        y_pred = coefficients * tes[echo]

        # compute the difference and get the integer multiple map
        int_map = np.round(
            (y_pred - unwrapped_data[mask, echo, t]) / (2 * np.pi)
        ).astype(int)

        # correct the data using the integer map
        unwrapped_data[mask, echo, t] += 2 * np.pi * int_map


def compute_field_map(
    unwrapped_mat: npt.NDArray,
    mag: list[nib.Nifti1Image],
    num_echos: int,
    tes_mat: npt.NDArray,
    frame_num: int,
) -> npt.NDArray:
    """Function for computing field map for a given frame.

    Parameters
    ----------
    unwrapped_mat : np.ndarray of shape (x, y, z, num_echos)
        Array of unwrapped phase data for a given frame
    mag : List[nib.NiftiImage] of shape (x, y, z, num_echos)
        List of magnitudes
    num_echos : int
        Number of echos
    tes_mat : npt.NDArray of shape (num_echos, 1)
        Echo times in a 2d matrix
    frame_num : int
        Frame number

    Returns
    -------
    b0 : np.ndarray of shape (x, y, z)
        Field map in Hertz.
    """
    logging.info(f"Computing field map for frame: {frame_num}")
    unwrapped_mat = unwrapped_mat.reshape(-1, num_echos).T
    mag_data = np.stack([m.dataobj[..., frame_num] for m in mag], axis=-1).astype(
        np.float32
    )
    weights = mag_data.reshape(-1, num_echos).T
    b0 = weighted_regression(tes_mat, unwrapped_mat, weights)[0].T.reshape(
        *mag_data.shape[:3]
    )
    b0 *= 1000 / (2 * np.pi)
    return b0


def compute_offset(
    echo_ind: int, w_mat: npt.NDArray, x_mat: npt.NDArray, y_mat: npt.NDArray
) -> int:
    """Method for computing the global mode offset for echoes > 1.

    Parameters
    ----------
    echo_ind : int
        Echo index
    w_mat : npt.NDArray of shape (num_echos, n_voxels)
        Weights
    x_mat : npt.NDArray of shape (num_echos, 1)
        tes in 2d matrix
    y_mat : npt.NDArray of shape (num_echos, n_voxels)
        Masked unwrapped data weighted by magnitude

    Returns
    -------
    best_offset : int
    """
    # fit the model to the up to previous echo
    coefficients, _ = weighted_regression(
        x_mat[:echo_ind], y_mat[:echo_ind], w_mat[:echo_ind]
    )

    # compute the predicted phase for the current echo
    y_pred = x_mat[echo_ind] * coefficients

    # compute the difference between the predicted phase and the unwrapped phase
    y_diff = y_pred - y_mat[echo_ind]

    # compute closest multiple of 2pi to the difference
    int_map = np.round(y_diff / (2 * np.pi)).astype(int)

    # compute the most often occuring multiple
    best_offset = mode(int_map, axis=0, keepdims=False).mode
    best_offset = cast(int, best_offset)

    return best_offset


def svd_filtering(
    field_maps: npt.NDArray,
    new_masks: npt.NDArray,
    voxel_size: float,
    n_frames: int,
    border_filt: tuple[int, int],
    svd_filt: int,
):
    """Apply an SVD-based filter to a 4D field map.

    Parameters
    ----------
    field_maps : np.ndarray of shape (x, y, z, n_frames)
        4D field map array.
        This array will be modified in place.
    new_masks : np.ndarray of shape (x, y, z, n_frames)
        Brain mask array. May be binary or may have 0s, 1s, and 2s.
    voxel_size : float
        Voxel size along the first dimension of the phase image, in millimeters.
        Does this assume isotropic voxels?
    n_frames : int
        Number of volumes in the run.
    border_filt : tuple of (int, int)
        Border filter.
    svd_filt : int
        SVD filter size.
    """
    if new_masks.max() == 2 and n_frames >= np.max(border_filt):
        logging.info("Performing spatial/temporal filtering of border voxels...")
        smoothed_field_maps = np.zeros(field_maps.shape, dtype=np.float32)

        # smooth by 4 mm kernel
        sigma = (4 / voxel_size) / 2.355
        for i_vol in range(field_maps.shape[-1]):
            smoothed_field_maps[..., i_vol] = gaussian_filter(
                field_maps[..., i_vol], sigma=sigma
            )

        # compute the union of all the masks
        union_mask = np.sum(new_masks, axis=-1) > 0

        # do temporal filtering of border voxels with SVD
        u, s, vt = np.linalg.svd(smoothed_field_maps[union_mask], full_matrices=False)

        # first pass of SVD filtering
        recon = np.dot(
            u[:, : border_filt[0]] * s[: border_filt[0]], vt[: border_filt[0], :]
        )
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon

        # set the border voxels in the field map to the recon values
        for i_vol in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i_vol] == 1, i_vol] = recon_img[
                new_masks[..., i_vol] == 1, i_vol
            ]

        # do second SVD filtering pass
        u, s, vt = np.linalg.svd(field_maps[union_mask], full_matrices=False)

        # second pass of SVD filtering
        recon = np.dot(
            u[:, : border_filt[1]] * s[: border_filt[1]], vt[: border_filt[1], :]
        )
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon

        # set the border voxels in the field map to the recon values
        for i_vol in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i_vol] == 1, i_vol] = recon_img[
                new_masks[..., i_vol] == 1, i_vol
            ]

    # use svd filter to denoise the field maps
    if n_frames >= svd_filt:
        logging.info("Denoising field maps with SVD...")
        logging.info(f"Keeping {svd_filt} components...")

        # compute the union of all the masks
        union_mask = np.sum(new_masks, axis=-1) > 0

        # compute SVD
        u, s, vt = np.linalg.svd(field_maps[union_mask], full_matrices=False)

        # only keep the first n_components components
        recon = np.dot(u[:, :svd_filt] * s[:svd_filt], vt[:svd_filt, :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon

        # set the voxel values in the mask to the recon values
        for i_vol in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i_vol] > 0, i_vol] = recon_img[
                new_masks[..., i_vol] > 0, i_vol
            ]


def unwrap_phases(
    phase: list[nib.Nifti1Image],
    mag: list[nib.Nifti1Image],
    tes: list[float] | tuple[float] | npt.NDArray[np.float32],
    mask: nib.Nifti1Image | SimpleNamespace | None = None,
    automask: bool = True,
    automask_dilation: int = 3,
    frames: list[int] | None = None,
    n_cpus: int = 4,
    debug: bool = False,
) -> tuple[list[nib.Nifti1Image], nib.Nifti1Image]:
    """Unwrap multi-echo phase per frame and enforce temporal consistency.

    Calls ROMEO via the warpkit C++ bindings. The returned unwrapped phases are
    useful outside of distortion correction (e.g. phase regression, T2*
    estimation). Pair with :func:`compute_field_maps` to reconstruct the
    native-space B0 field map.

    Parameters
    ----------
    phase : list[nib.Nifti1Image]
        Phase images, one per echo. Each may be 3D or 4D.
    mag : list[nib.Nifti1Image]
        Magnitude images matched to ``phase``.
    tes : list[float] | tuple[float] | npt.NDArray
        Echo times in milliseconds, one per echo.
    mask : nib.Nifti1Image, optional
        Boolean mask. Ignored when ``automask`` is True.
    automask : bool, optional
        Auto-generate the mask, by default True.
    automask_dilation : int, optional
        Dilation iterations for the auto mask, by default 3.
    frames : list[int], optional
        Subset of frame indices to process, by default None (all frames).
    n_cpus : int, optional
        CPU parallelism for the unwrap loop, by default 4.
    debug : bool, optional
        Skip the temporal consistency pass and dump intermediate files, by
        default False.

    Returns
    -------
    list[nib.Nifti1Image]
        Per-echo unwrapped phase images (radians), each of shape
        ``(x, y, z, n_frames)``.
    nib.Nifti1Image
        Per-frame masks, shape ``(x, y, z, n_frames)``, dtype int8. 0 = outside,
        1 = brain core, 2 = dilated border (consumed by SVD filtering in
        :func:`compute_field_maps`).
    """
    # check tes if < 0.1, tell user they probably need to convert to ms
    if np.min(tes) < 0.1:
        logging.warning(
            "WARNING: tes are unusually small. Your inputs may be incorrect. Did you forget to convert to ms?"
        )

    # convert tes to np array
    tes = cast(npt.NDArray[np.float32], np.array(tes))

    # make sure affines/shapes are all correct
    for img in (*phase, *mag):
        if img.affine is None:
            raise ValueError("All input images must have a non-None affine.")
    for p1, m1 in zip(phase, mag, strict=True):
        p1_aff = cast(np.ndarray, p1.affine)
        m1_aff = cast(np.ndarray, m1.affine)
        for p2, m2 in zip(phase, mag, strict=True):
            p2_aff = cast(np.ndarray, p2.affine)
            m2_aff = cast(np.ndarray, m2.affine)
            if not (
                np.allclose(p1_aff, p2_aff, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.shape, p2.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(m1_aff, m2_aff, rtol=1e-3, atol=1e-3)
                and np.allclose(m1.shape, m2.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(p1_aff, m1_aff, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.shape, m1.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(p2_aff, m2_aff, rtol=1e-3, atol=1e-3)
                and np.allclose(p2.shape, m2.shape, rtol=1e-3, atol=1e-3)
            ):
                raise ValueError("Affines/Shapes of images do not all match.")

    # check if data is 4D or 3D
    if len(phase[0].shape) == 3:
        # convert data to 4D
        phase = [
            nib.Nifti1Image(p.get_fdata()[..., np.newaxis], p.affine, p.header)
            for p in phase
        ]
        mag = [
            nib.Nifti1Image(m.get_fdata()[..., np.newaxis], m.affine, m.header)
            for m in mag
        ]
        if frames is None:
            frames = [0]
    elif len(phase[0].shape) == 4:
        # if frames is None, set it to all frames
        if frames is None:
            frames = list(range(phase[0].shape[-1]))
    else:
        raise ValueError("Data must be 3D or 4D.")
    # frames should be a list at this point
    frames = cast(list[int], frames)

    # check echo times = number of mag and phase images
    if len(tes) != len(phase) or len(tes) != len(mag):
        raise ValueError(
            "Number of echo times must equal number of mag and phase images."
        )

    # allocate space for unwrapped phases and per-frame masks
    unwrapped = np.zeros((*phase[0].shape[:3], len(tes), len(frames)), dtype=np.float32)
    new_masks = np.zeros((*mag[0].shape[:3], len(frames)), dtype=np.int8)

    # FOR DEBUGGING
    if debug:
        global affine
        affine = phase[0].affine
        global header
        header = phase[0].header

    # allocate mask if needed
    if not mask:
        mask = SimpleNamespace()
        if (
            automask
        ):  # if we are automasking use a fake array that plays nice with the logic
            mask.dataobj = np.ones((1, 1, 1, phase[0].shape[-1]))
        else:
            mask.dataobj = np.ones(phase[0].shape)

    # write a function to iterate over each frame for phase unwrapping
    def phase_iterator(phase, mag, tes, mask, frames, automask, automask_dilation):
        # note that I separate out idx and frame_idx for the case when the user wants to process a subset of
        # non-contiguous frames
        # this will always reindex the frames to be contiguous however
        # e.g. if the user passes in frames [3, 5, 13] it will be reindexed to [0, 1, 2]

        # estimate the min and max phase values, we assume the phase ranges from -pi to pi
        min_phases = []
        max_phases = []
        for phase_echo in phase:
            # only look at the first frame
            phase_data = phase_echo.dataobj[..., frames[0]]
            # get the min and max phase value
            min_phases.append(phase_data.min())
            max_phases.append(phase_data.max())
        min_phase = mode(min_phases, keepdims=False).mode
        max_phase = mode(max_phases, keepdims=False).mode
        logging.info("Estimated min phase: %f", min_phase)
        logging.info("Estimated max phase: %f", max_phase)

        for idx, frame_idx in enumerate(frames):
            # get the phase and magnitude data from each echo
            phase_data: npt.NDArray[np.float32] = rescale_phase(
                np.stack([p.dataobj[..., frame_idx] for p in phase], axis=-1),
                min=min_phase,
                max=max_phase,
            ).astype(np.float32)
            mag_data: npt.NDArray[np.float32] = np.stack(
                [m.dataobj[..., frame_idx] for m in mag], axis=-1
            ).astype(np.float32)
            mask_data = cast(
                npt.NDArray[np.bool_], mask.dataobj[..., frame_idx].astype(bool)
            )
            tes = tes.astype(np.float32)
            yield (
                phase_data,
                mag_data,
                tes,
                mask_data,
                automask,
                automask_dilation,
                idx,
                debug,
            )

    def save_unwrapped_and_mask(idx, result):
        # get the unwrapped image
        logging.info(f"Collecting frame: {idx}")
        # store in the captured unwrapped and new_mask_data arrays
        unwrapped[..., idx], new_masks[..., idx] = result

    # unwrap the phase of each frame. romeo_unwrap* releases the GIL around the
    # C++ kernel, so threads parallelize it as well as processes while avoiding
    # per-frame array pickling and per-worker memory copies.
    run_executor(
        ncpus=n_cpus,
        type="thread",
        fn=unwrap_phase,
        iterator=phase_iterator(
            phase, mag, tes, mask, frames, automask, automask_dilation
        ),
        post_fn=save_unwrapped_and_mask,
    )

    def temporal_consistency_iterator(unwrapped, tes, mag, frames, masks):
        # Make a copy of the first echo data
        unwrapped_echo_1 = unwrapped[..., 0, :].copy()

        logging.info("Computing temporal consistency...")
        for t, frame_idx in enumerate(frames):
            yield (
                unwrapped,
                unwrapped_echo_1,
                tes,
                mag,
                t,
                frame_idx,
                cast(npt.NDArray, masks),
            )

    def post_temporal_consistency_check(idx, result):
        logging.info(f"Temporal consistency check for frame {idx} complete.")

    if not debug:
        run_executor(
            ncpus=n_cpus,
            type="thread",
            fn=check_temporal_consistency_corr,
            iterator=temporal_consistency_iterator(
                unwrapped, tes, mag, frames, new_masks
            ),
            post_fn=post_temporal_consistency_check,
        )

    # Save out unwrapped phase for debugging
    if debug:
        logging.info("Saving unwrapped phase images...")
        for i_vol in range(unwrapped.shape[-2]):
            nib.Nifti1Image(
                unwrapped[:, :, :, i_vol, :],
                phase[0].affine,
                phase[0].header,
            ).to_filename(f"phase{i_vol}.nii")

        logging.info("Saving masks..")
        nib.Nifti1Image(new_masks, phase[0].affine, phase[0].header).to_filename(
            "masks.nii"
        )

    # split the (x, y, z, echo, frame) array into a list of 4D Niftis (one per echo)
    unwrapped_imgs = [
        nib.Nifti1Image(unwrapped[:, :, :, i_echo, :], phase[0].affine, phase[0].header)
        for i_echo in range(unwrapped.shape[-2])
    ]
    masks_img = nib.Nifti1Image(new_masks, mag[0].affine, mag[0].header)
    return unwrapped_imgs, masks_img


def compute_field_maps(
    unwrapped: list[nib.Nifti1Image],
    masks: nib.Nifti1Image,
    mag: list[nib.Nifti1Image],
    tes: list[float] | tuple[float] | npt.NDArray[np.float32],
    border_filt: tuple[int, int] = (1, 5),
    svd_filt: int = 10,
    n_cpus: int = 4,
) -> nib.Nifti1Image:
    """Compute native-space B0 field maps from unwrapped phases.

    Performs the second half of MEDIC: weighted echo regression per frame to
    yield a field map in Hz, followed by border-aware + global SVD filtering
    using the per-frame masks from :func:`unwrap_phases`.

    Parameters
    ----------
    unwrapped : list[nib.Nifti1Image]
        Per-echo unwrapped phase images (radians), each shape
        ``(x, y, z, n_frames)``. Order must match ``mag`` and ``tes``.
    masks : nib.Nifti1Image
        Per-frame masks, shape ``(x, y, z, n_frames)``, with values 0/1/2 as
        produced by :func:`unwrap_phases`.
    mag : list[nib.Nifti1Image]
        Magnitude images, used as regression weights.
    tes : list[float] | tuple[float] | npt.NDArray
        Echo times in milliseconds.
    border_filt : tuple[int, int], optional
        SVD components for the two-pass border filter, by default ``(1, 5)``.
    svd_filt : int, optional
        SVD components for global denoising, by default 10.
    n_cpus : int, optional
        CPU parallelism for the field map loop, by default 4.

    Returns
    -------
    nib.Nifti1Image
        Native-space field map in Hz, shape ``(x, y, z, n_frames)``.
    """
    if len(unwrapped) != len(mag) or len(unwrapped) != len(tes):
        raise ValueError(
            "Number of unwrapped phase images, magnitude images, and echo "
            "times must all match."
        )

    tes = cast(npt.NDArray[np.float32], np.array(tes, dtype=np.float32))

    # stack per-echo 4D unwrapped images into (x, y, z, n_echoes, n_frames)
    unwrapped_arr = np.stack(
        [np.asarray(u.dataobj, dtype=np.float32) for u in unwrapped], axis=-2
    )
    new_masks = np.asarray(masks.dataobj, dtype=np.int8)
    n_frames = unwrapped_arr.shape[-1]

    if (
        new_masks.ndim != 4
        or new_masks.shape[:3] != unwrapped_arr.shape[:3]
        or new_masks.shape[-1] != n_frames
    ):
        raise ValueError(
            "masks must have shape (x, y, z, n_frames) matching the spatial "
            "dimensions and frame count of the unwrapped data; got masks shape "
            f"{new_masks.shape} and expected "
            f"({unwrapped_arr.shape[0]}, {unwrapped_arr.shape[1]}, "
            f"{unwrapped_arr.shape[2]}, {n_frames})."
        )

    # allocate output
    field_maps = np.zeros((*unwrapped_arr.shape[:3], n_frames), dtype=np.float32)

    def field_map_iterator(unwrapped_arr, mag, tes):
        logging.info("Running field map computation...")
        tes_mat = tes[:, np.newaxis]
        for frame_num in range(unwrapped_arr.shape[-1]):
            yield (unwrapped_arr[..., frame_num], mag, tes.shape[0], tes_mat, frame_num)

    def post_field_map(idx, result):
        logging.info(f"Field map computation for frame {idx} complete.")
        field_maps[..., idx] = result

    run_executor(
        ncpus=n_cpus,
        type="thread",
        fn=compute_field_map,
        iterator=field_map_iterator(unwrapped_arr, mag, tes),
        post_fn=post_field_map,
    )

    # border-aware + global SVD filtering of the field maps
    svd_filtering(
        field_maps,
        new_masks,
        unwrapped[0].header.get_zooms()[0],  # type: ignore
        n_frames,
        border_filt,
        svd_filt,
    )

    return nib.Nifti1Image(field_maps, unwrapped[0].affine, unwrapped[0].header)


def unwrap_and_compute_field_maps(
    phase: list[nib.Nifti1Image],
    mag: list[nib.Nifti1Image],
    tes: list[float] | tuple[float] | npt.NDArray[np.float32],
    mask: nib.Nifti1Image | SimpleNamespace | None = None,
    automask: bool = True,
    automask_dilation: int = 3,
    border_size: int = 5,
    border_filt: tuple[int, int] = (1, 5),
    svd_filt: int = 10,
    frames: list[int] | None = None,
    n_cpus: int = 4,
    debug: bool = False,
) -> nib.Nifti1Image:
    """Unwrap phase and compute native-space field maps in a single call.

    Thin wrapper around :func:`unwrap_phases` followed by
    :func:`compute_field_maps`. See those functions for parameter and return
    semantics. ``border_size`` is accepted for backwards compatibility but is
    currently unused. Returns the native-space field map in Hz.
    """
    del border_size  # accepted for back-compat; not consumed by either stage
    unwrapped_imgs, masks_img = unwrap_phases(
        phase,
        mag,
        tes,
        mask=mask,
        automask=automask,
        automask_dilation=automask_dilation,
        frames=frames,
        n_cpus=n_cpus,
        debug=debug,
    )
    return compute_field_maps(
        unwrapped_imgs,
        masks_img,
        mag,
        tes,
        border_filt=border_filt,
        svd_filt=svd_filt,
        n_cpus=n_cpus,
    )
