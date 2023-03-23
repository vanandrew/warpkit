import sys
import signal
import logging
import traceback
from types import SimpleNamespace
from typing import cast, List, Tuple, Union
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import nibabel as nib
import numpy as np
import numpy.typing as npt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import (
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
    binary_fill_holes,
)
from scipy.stats import mode
from warpkit.model import weighted_regression
from .utilities import rescale_phase
from . import JuliaContext as _JuliaContext  # type: ignore


def initialize_julia_context():
    """Initialize a Julia context in the global variable JULIA."""

    # This should only be initialized once
    global JULIA
    JULIA = _JuliaContext()

    # Set the process's signal handler to ignore SIGINT
    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])


def create_brain_mask(mag_shortest: npt.NDArray[np.float64], extra_dilation: int = 6) -> npt.NDArray[np.bool8]:
    """Create a quick brain mask for a single frame.

    Parameters
    ----------
    mag_shortest : npt.NDArray[np.float64]
        Magnitude data with the shortest echo time
    extra_dilation : int
        Number of extra dilations to perform

    Returns
    -------
    npt.NDArray[np.bool8]
        Mask of voxels to use for unwrapping
    """
    # create structuring element
    strel = generate_binary_structure(3, 2)

    # get the otsu threshold
    threshold = threshold_otsu(mag_shortest)
    mask_data = mag_shortest > threshold
    mask_data = cast(npt.NDArray[np.float64], binary_fill_holes(mask_data, strel))

    # erode mask
    mask_data = binary_erosion(mask_data, structure=strel, iterations=2, border_value=1)

    # get largest connected component
    labelled_mask = label(mask_data)
    props = regionprops(labelled_mask)
    sorted_props = sorted(props, key=lambda x: x.area, reverse=True)
    mask_data = labelled_mask == sorted_props[0].label

    # dilate the mask
    mask_data = binary_dilation(mask_data, structure=strel, iterations=2)

    # extra dilation to get areas on the edge of the brain
    mask_data = binary_dilation(mask_data, structure=strel, iterations=extra_dilation)

    # since we can't have a completely empty mask, set all zeros to ones
    # if the mask is all empty
    if np.all(np.isclose(mask_data, 0)):
        mask_data = np.ones(mask_data.shape)

    # return the mask
    return mask_data.astype(np.bool8)


def unwrap_phase(
    phase_data: npt.NDArray[np.float64],
    mag_data: npt.NDArray[np.float64],
    TEs: npt.NDArray[np.float64],
    mask_data: npt.NDArray[np.bool8],
    automask: bool = True,
    correct_global: bool = True,
    idx: Union[int, None] = None,
) -> npt.NDArray[np.float64]:
    """Unwraps the phase for a single frame of ME-EPI

    Parameters
    ----------
    phase_data : npt.NDArray[np.float64]
        Single frame of phase data with shape (x, y, z, echo)
    mag_data : npt.NDArray[np.float64]
        Single frame of magnitude data with shape (x, y, z, echo)
    TEs : npt.NDArray[np.float64]
        Echo times associated with each phase
    mask_data : npt.NDArray[np.bool8]
        Mask of voxels to use for unwrapping
    automask : bool, optional
        Automatically compute a mask, by default True
    correct_global : bool, optional
        Corrects global n2π offset, by default True
    idx : int, optional
        Index of the frame being processed for verbosity, by default None

    Returns
    -------
    npt.NDArray[np.float64]
        unwrapped phase in radians
    """
    # use global Julia Context
    global JULIA

    if idx is not None:
        logging.info(f"Processing frame: {idx}")

    # if automask is True, generate a mask for the frame, instead of using mask_data
    if automask:
        # get the index with the shortest echo time
        echo_idx = np.argmin(TEs)
        mag_shortest = mag_data[..., echo_idx]

        # create the mask
        mask_data = create_brain_mask(mag_shortest)

    # Do MCPC-3D-S algo to compute phase offset
    signal_diff = mag_data[..., 0] * mag_data[..., 1] * np.exp(1j * (phase_data[..., 1] - phase_data[..., 0]))
    mag_diff = np.abs(signal_diff)
    phase_diff = np.angle(signal_diff)
    unwrapped_diff = JULIA.romeo_unwrap3D(
        phase=phase_diff,
        weights="romeo",
        mag=mag_diff,
        mask=mask_data,
    )
    # compute the phase offset
    phase_offset = np.angle(np.exp(1j * (phase_data[..., 0] - ((TEs[0] * unwrapped_diff) / (TEs[1] - TEs[0])))))
    # remove phase offset from data
    phase_data -= phase_offset[..., np.newaxis]

    # unwrap the phase data
    unwrapped = JULIA.romeo_unwrap4D(
        phase=phase_data,
        TEs=TEs,
        weights="romeo",
        mag=mag_data,
        mask=mask_data,
        correct_global=False,
        maxseeds=1,
        merge_regions=False,
        correct_regions=False,
    )
    # global mode correction
    # this computes the global mode offset for the first echo then tries to find the offset
    # that minimizes the residuals for each subsequent echo
    # this seems to be more robust than romeo's global offset correction
    if correct_global:
        unwrapped -= (
            mode(np.round(unwrapped[mask_data, 0] / (2 * np.pi)).astype(int), axis=0, keepdims=False).mode * 2 * np.pi
        )

        # for each of these matrices TEs are on rows, voxels are columns
        # get design matrix
        X = TEs[:, np.newaxis]

        # get magnitude weight matrix
        W = mag_data[mask_data, :].T
        # loop over each index past 2nd echo

        # best_offset = compute_offset(i, W, X, Y)

        for i in range(1, TEs.shape[0]):
            # get matrix with the masked unwrapped data (weighted by magnitude)
            Y = unwrapped[mask_data, :].T

            # Compute offset by method that minimizes residuals
            best_offset = compute_offset_residual(i, W, X, Y)
            unwrapped[mask_data, i] += 2 * np.pi * best_offset

    # return the unwrapped data
    return unwrapped


def consecutive(data, idx):
    # get consecutive groups
    groups = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
    # only return group with idx
    return [g for g in groups if idx in g][0]


def check_temporal_consistency(
    unwrapped_data: npt.NDArray,
    motion_params: npt.NDArray,
    weights: List[nib.Nifti1Image],
    TEs,
    rd_threshold: float = 0.5,
) -> npt.NDArray:
    """Ensures phase unwrapping solutions are temporally consistent

    Parameters
    ----------
    unwrapped_data : npt.NDArray
        unwrapped phase data, where last column is time, and second to last column are the echoes
    motion_params : npt.NDArray
        motion parameters, ordered as x, y, z, rot_x, rot_y, rot_z, note that rotations should be preconverted
        to mm
    weights : List[nib.Nifti1Image]
        weights for regression model
    TEs : npt.NDArray
        echo times
    rd_threshold : float
        relative displacement threshold. By default 0.5

    Returns
    -------
    npt.NDArray
        fixed unwrapped phase data
    """
    # look at the first echo
    unwrapped_echo_1 = unwrapped_data[..., 0, :].copy()

    # loop over each frame of the data
    for t in range(unwrapped_echo_1.shape[-1]):
        logging.info("Computing temporal consistency check for frame: %d" % t)
        # compute the relative displacements to the current frame
        # first get the motion parameters for the current frame
        current_frame_motion_params = motion_params[t][np.newaxis, :]

        # now get the difference of each motion params to this frame
        motion_params_diff = motion_params - current_frame_motion_params

        # now compute the relative displacement by sum of the absolute values
        RD = np.sum(np.abs(motion_params_diff), axis=1)

        # threhold the RD
        tmask = RD < rd_threshold

        # get indices of mask
        indices = np.where(tmask)[0]

        # get the consecutive group
        c_indices = consecutive(indices, t)
        # for each frame compute the mean value along the time axis in the contiguous group
        mean_voxels = np.mean(unwrapped_echo_1[..., c_indices], axis=-1)

        # for this frame figure out the integer multiple that minimizes the value to the mean voxel
        int_map = np.round((mean_voxels - unwrapped_echo_1[..., t]) / (2 * np.pi)).astype(int)

        # correct the data using the integer map
        unwrapped_data[..., 0, t] += 2 * np.pi * int_map

        # TODO: this is dangerous, align use of frames vs natural index
        # format weight matrix
        weights_mat = (
            np.stack([m.dataobj[..., t] for m in weights], axis=-1).astype(np.float64).reshape(-1, TEs.shape[0]).T
        )

        # fit subsequenct echos to the recursive weighted linear regression from the first echo
        for echo in range(1, unwrapped_data.shape[-2]):
            # form design matrix
            X = TEs[:echo, np.newaxis]

            # form response matrix
            Y = unwrapped_data[..., :echo, t].reshape(-1, echo).T

            # fit model to data
            coefficients, _ = weighted_regression(X, Y, weights_mat[:echo])

            # get the predicted values for this echo
            Y_pred = coefficients * TEs[echo]

            # compute the difference and get the integer multiple map
            int_map = (
                np.round((Y_pred - unwrapped_data[..., echo, t].reshape(-1).T) / (2 * np.pi))
                .astype(int)
                .T.reshape(*unwrapped_data.shape[:3])
            )

            # correct the data using the integer map
            unwrapped_data[..., echo, t] += 2 * np.pi * int_map

    # return the fixed data
    return unwrapped_data


def start_unwrap_process(
    unwrapped: npt.NDArray,
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float], npt.NDArray[np.float64]],
    mask: Union[nib.Nifti1Image, SimpleNamespace, None] = None,
    automask: bool = True,
    correct_global: bool = True,
    frames: Union[List[int], None] = None,
    n_cpus: int = 4,
):
    """
    Parameters
    ----------
    unwrapped: npt.NDArray
        Array to store unwrapped data
    phase : List[nib.Nifti1Image]
        Phases to unwrap
    mag : List[nib.Nifti1Image]
        Magnitudes associated with each phase
    TEs : Union[List[float], Tuple[float], npt.NDArray[np.float64]]
        Echo times associated with each phase (in ms)
    mask : nib.Nifti1Image, optional
        Boolean mask, by default None
    automask : bool, optional
        Automatically generate a mask (ignore mask option), by default True
    correct_global : bool, optional
        Corrects global n2π offsets, by default True
    frames : List[int], optional
        Only process these frame indices, by default None (which means all frames)
    n_cpus : int, optional
        Number of CPUs to use, by default 4
    """

    # if multiprocessing is not enabled, run serially
    if n_cpus == 1:
        logging.info("Running unwrap phase serially")
        initialize_julia_context()
        for idx in frames:
            # get the phase and magnitude data from each echo
            phase_data: npt.NDArray[np.float64] = rescale_phase(
                np.stack([p.dataobj[..., idx] for p in phase], axis=-1)
            ).astype(np.float64)
            mag_data: npt.NDArray[np.float64] = np.stack([m.dataobj[..., idx] for m in mag], axis=-1).astype(np.float64)

            mask_data = cast(npt.NDArray[np.bool8], mask.dataobj[..., idx].astype(bool))

            unwrapped[..., idx] = unwrap_phase(
                phase_data,
                mag_data,
                TEs,
                mask_data,
                automask,
                correct_global,
                idx,
            )
    # if multiprocessing is enabled, run in parallel using a process pool
    else:
        try:
            logging.info(f"Running unwrap phase in parallel with {n_cpus} processes")

            # use a process pool to speed up the computation
            executor = ProcessPoolExecutor(max_workers=n_cpus, initializer=initialize_julia_context)

            # set error flag for process pool
            PROCESS_POOL_ERROR = False

            # collect futures in a dictionary, where the value is the frame index
            futures = dict()
            # loop over the total number of frames
            for idx in frames:
                # get the phase and magnitude data from each echo
                phase_data: npt.NDArray[np.float64] = rescale_phase(
                    np.stack([p.dataobj[..., idx] for p in phase], axis=-1)
                ).astype(np.float64)
                mag_data: npt.NDArray[np.float64] = np.stack([m.dataobj[..., idx] for m in mag], axis=-1).astype(
                    np.float64
                )

                mask_data = cast(npt.NDArray[np.bool8], mask.dataobj[..., idx].astype(bool))

                # submit field map computation to the process pool
                futures[
                    executor.submit(
                        unwrap_phase,
                        phase_data,
                        mag_data,
                        TEs,
                        mask_data,
                        automask,
                        correct_global,
                        idx,
                    )
                ] = idx

            # loop over the futures
            for future in as_completed(futures):
                # get the frame index
                idx = futures[future]
                # get the unwrapped image
                logging.info(f"Collecting frame: {idx}")
                unwrapped[..., idx] = future.result()
        except KeyboardInterrupt:
            # on keyboard interrupt, just kill all processes in the executor pool
            for proc in executor._processes.values():
                proc.kill()
            # set global error flag
            PROCESS_POOL_ERROR = True
        except Exception:
            traceback.print_exc()
            # set global error flag
            PROCESS_POOL_ERROR = True
        finally:
            if PROCESS_POOL_ERROR:
                sys.exit(1)
            # close the executor
            else:
                executor.shutdown()

    return unwrapped


def start_fieldmap_process(
    field_maps: npt.NDArray,
    unwrapped: npt.NDArray,
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float], npt.NDArray[np.float64]],
    n_cpus: int = 4,
) -> npt.NDArray:
    """

    Parameters
    ----------
    field_maps : npt.NDArray
        Array to store field maps
    unwrapped : npt.NDArray
        Array of unwrapped phase data
    mag : List[nib.Nifti1Image]
        Magnitudes associated with each phase
    TEs : Union[List[float], Tuple[float], npt.NDArray[np.float64]]
        Echo times associated with each phase (in ms)
    n_cpus : int, optional
        Number of CPUs to use, by default 4
    """

    TEs_mat = TEs[:, np.newaxis]

    # If multiprocessing is not enabled, run serially
    if n_cpus == 1:
        logging.info("Running field map computation serially")
        for frame_num in range(unwrapped.shape[-1]):
            field_maps[..., frame_num] = compute_field_map(
                unwrapped[..., frame_num], mag, TEs.shape[0], TEs_mat, frame_num
            )
    # If multiprocessing is enabled, run in parallel using a process pool
    else:
        logging.info(f"Running field map computation in parallel with {n_cpus} processes")

        with ProcessPoolExecutor(max_workers=n_cpus) as fieldmap_executor:
            # collect futures in a dictionary, where the value is the frame index
            fieldmap_futures = dict()
            # loop over the total number of frames
            for frame_num in range(unwrapped.shape[-1]):

                fieldmap_futures[
                    fieldmap_executor.submit(
                        compute_field_map, unwrapped[..., frame_num], mag, TEs.shape[0], TEs_mat, frame_num
                    )
                ] = frame_num

            # loop over the futures
            for fieldmap_future in as_completed(fieldmap_futures):
                # get the frame index
                frame_num = fieldmap_futures[fieldmap_future]
                logging.info(f"Loading frame: {frame_num}")
                field_maps[..., frame_num] = fieldmap_future.result()

    return field_maps


def unwrap_and_compute_field_maps(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float], npt.NDArray[np.float64]],
    mask: Union[nib.Nifti1Image, SimpleNamespace, None] = None,
    automask: bool = True,
    correct_global: bool = True,
    frames: Union[List[int], None] = None,
    motion_params: Union[npt.NDArray, None] = None,
    n_cpus: int = 4,
) -> nib.Nifti1Image:
    """Unwrap phase of data weighted by magnitude data and compute field maps. This makes a call
    to the ROMEO phase unwrapping algorithm for each frame. To learn more about ROMEO, see this paper:

    Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K., Robinson, S.D., 2020.
    Phase Unwrapping with a Rapid Opensource Minimum Spanning TreE AlgOrithm (ROMEO).
    Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.28563

    Parameters
    ----------
    phase : List[nib.Nifti1Image]
        Phases to unwrap
    mag : List[nib.Nifti1Image]
        Magnitudes associated with each phase
    TEs : Union[List[float], Tuple[float], npt.NDArray[np.float64]]
        Echo times associated with each phase (in ms)
    mask : nib.Nifti1Image, optional
        Boolean mask, by default None
    automask : bool, optional
        Automatically generate a mask (ignore mask option), by default True
    correct_global : bool, optional
        Corrects global n2π offsets, by default True
    frames : List[int], optional
        Only process these frame indices, by default None (which means all frames)
    n_cpus : int, optional
        Number of CPUs to use, by default 4

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz
    """
    # check TEs if < 0.1, tell user they probably need to convert to ms
    if np.min(TEs) < 0.1:
        logging.warning(
            "WARNING: TEs are unusually small. Your inputs may be incorrect. Did you forget to convert to ms?"
        )

    # convert TEs to np array
    TEs = cast(npt.NDArray[np.float64], np.array(TEs))

    # make sure affines/shapes are all correct
    for p1, m1 in zip(phase, mag):
        for p2, m2 in zip(phase, mag):
            if not (
                np.allclose(p1.affine, p2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.shape, p2.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(m1.affine, m2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(m1.shape, m2.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.affine, m1.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.shape, m1.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(p2.affine, m2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(p2.shape, m2.shape, rtol=1e-3, atol=1e-3)
            ):
                raise ValueError("Affines/Shapes of images do not all match.")

    # check if data is 4D or 3D
    if len(phase[0].shape) == 3:
        # set total number of frames to 1
        n_frames = 1
        # convert data to 4D
        phase = [nib.Nifti1Image(p.get_fdata()[..., np.newaxis], p.affine, p.header) for p in phase]
        mag = [nib.Nifti1Image(m.get_fdata()[..., np.newaxis], m.affine, m.header) for m in mag]
    elif len(phase[0].shape) == 4:
        # if frames is None, set it to all frames
        if frames is None:
            frames = list(range(phase[0].shape[-1]))
        # get the total number of frames
        n_frames = len(frames)
    else:
        raise ValueError("Data must be 3D or 4D.")
    # frames should be a list at this point
    frames = cast(List[int], frames)

    # check echo times = number of mag and phase images
    if len(TEs) != len(phase) or len(TEs) != len(mag):
        raise ValueError("Number of echo times must equal number of mag and phase images.")

    # # allocate space for field maps and unwrapped
    field_maps = np.zeros((*phase[0].shape[:3], n_frames))
    unwrapped = np.zeros((*phase[0].shape[:3], len(TEs), n_frames))

    # allocate mask if needed
    if not mask:
        mask = SimpleNamespace()
        mask.dataobj = np.ones((*phase[0].shape[:3], n_frames))

    # load in unwrapped image
    unwrapped = start_unwrap_process(unwrapped, phase, mag, TEs, mask, automask, correct_global, frames, n_cpus)

    # check temporal consistency to unwrapped phase
    # if motion parameters passed in
    if motion_params is not None:
        unwrapped = check_temporal_consistency(
            unwrapped,
            motion_params[frames],
            mag,
            TEs,
            0.5,
        )

    # # FOR DEBUGGING
    # TODO: create flag for this, default false
    # # save out unwrapped phase
    # logging.info("Saving unwrapped phase images...")
    # for i in range(unwrapped.shape[-2]):
    #     nib.Nifti1Image(
    #       unwrapped[:, :, :, i, :], phase[0].affine, phase[0].header).to_filename(f"unwrapped_{i}.nii.gz")
    # for i in range(unwrapped.shape[-2]):
    #     nib.Nifti1Image(mean_unwrapped[:, :, :, i, :], phase[0].affine, phase[0].header).to_filename(
    #         f"mean_unwrapped_{i}.nii.gz"
    #     )

    # compute field maps on temporally consistent unwrapped phase
    field_maps = start_fieldmap_process(field_maps, unwrapped, mag, TEs, n_cpus)

    # return the field map as a nifti image
    return nib.Nifti1Image(field_maps[..., frames], phase[0].affine, phase[0].header)


def compute_field_map(
    unwrapped_mat: npt.NDArray,
    mag: List[nib.Nifti1Image],
    num_echos: int,
    TEs_mat: Union[List[float], Tuple[float], npt.NDArray[np.float64]],
    frame_num: int,
) -> npt.NDArray:

    """
    Function for computing field map for a given frame

    Parameters
    ----------

    unwrapped_mat: np.ndarray
        Array of unwrapped phase data for a given frame
    mag: List[nib.NiftiImage]
        List of magnitudes
    num_echos: int
        Number of echos
    TEs_mat: Union[List[float], Tuple[float], np.ndarray]
        Echo times in a 2d matrix
    frame_num: int
        Frame number

    Returns
    -------
    B0: np.ndarray

    """

    logging.info(f"Computing field map for frame: {frame_num}")
    unwrapped_mat = unwrapped_mat.reshape(-1, num_echos).T
    mag_data = np.stack([m.dataobj[..., frame_num] for m in mag], axis=-1).astype(np.float64)
    weights = mag_data.reshape(-1, num_echos).T
    B0 = weighted_regression(TEs_mat, unwrapped_mat, weights)[0].T.reshape(*mag_data.shape[:3])
    B0 *= 1000 / (2 * np.pi)

    return B0


def compute_offset(
    echo_ind: int, W: npt.NDArray[np.float64], X: Union[List[float], Tuple[float], np.ndarray], Y: npt.NDArray
) -> int:

    """
    Method for computing the global mode offset without looping

    echo_ind: int
        Echo index
    W: np.ndarray
        Weights
    X: Union[List[float], Tuple[float], np.ndarray]
        TEs in 2d matrix
    Y: np.ndarray
        Masked unwrapped data weighted by magnitude

    Returns
    -------
    best_offset: int
    """

    # TODO: can do this without for loop by computing the 2pi multiple on the difference of the
    # estimated linear phase at a TE and the computed unwrapped phase at a TE

    # fit the model to the up to previous echo
    coefficients, _ = weighted_regression(X[:echo_ind], Y[:echo_ind], W[:echo_ind])

    # compute the predicted phase for the current echo
    Y_pred = X[echo_ind] * coefficients

    # compute the difference between the predicted phase and the unwrapped phase
    Y_diff = Y_pred - Y[echo_ind]

    # compute closest multiple of 2pi to the difference
    int_map = np.round(Y_diff / (2 * np.pi)).astype(int)

    # compute the most often occuring multiple
    best_offset = mode(int_map, axis=0, keepdims=False).mode
    best_offset = cast(int, best_offset)

    return best_offset


def compute_offset_residual(
    echo_ind: int, W: npt.NDArray[np.float64], X: Union[List[float], Tuple[float], np.ndarray], Y: npt.NDArray
) -> int:

    """
    Method for computing the offset by minimizing the residual

    echo_ind: int
        Echo index
    W: np.ndarray
        Weights
    X: Union[List[float], Tuple[float], np.ndarray]
        TEs in 2d matrix
    Y: np.ndarray
        Masked unwrapped data weighted by magnitude

    Returns
    -------
    best_offset: int
    """

    offsets = list(range(-10, 11))
    # create variables to store best residual and best offset
    best_residual = None
    best_offset = None
    for multiple in offsets:
        # get the ordinate matrix for these echoes
        Y_copy = Y.copy()[: echo_ind + 1, :]
        # add offset
        Y_copy[echo_ind, :] += 2 * np.pi * multiple

        # fit model and compute residuals
        _, residuals = weighted_regression(X[: echo_ind + 1], Y_copy, W[: echo_ind + 1])

        # get summary stat of residuals
        summary_residual = residuals.mean()

        # if no best residual, then just initialize it here
        if best_residual is None:
            best_residual = summary_residual
            best_offset = multiple
        else:  # else we store the residual + offset if it is better
            if summary_residual < best_residual:
                best_residual = summary_residual
                best_offset = multiple
    best_offset = cast(int, best_offset)

    return best_offset
