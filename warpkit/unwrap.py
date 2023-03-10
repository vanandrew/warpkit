import sys
import signal
import logging
from types import SimpleNamespace
from typing import cast, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import nibabel as nib
import numpy as np
import numpy.typing as npt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import ball
from scipy.ndimage import (
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
    binary_closing,
    binary_fill_holes,
    convolve,
)
from scipy.stats import mode
from warpkit.model import weighted_regression
from .utilities import rescale_phase, normalize
from . import JuliaContext as _JuliaContext  # type: ignore


def initialize_julia_context():
    """Initialize a Julia context in the global variable JULIA."""

    # This should only be initialized once
    global JULIA
    JULIA = _JuliaContext()

    # Set the process's signal handler to ignore SIGINT
    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])


def create_brain_mask(mag_shortest: npt.NDArray[np.float64]) -> npt.NDArray[np.bool8]:
    """Create a quick brain mask for a single frame.

    Parameters
    ----------
    mag_shortest : npt.NDArray[np.float64]
        Magnitude data with the shortest echo time

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

    # since we can't have a completely empty mask, set all zeros to ones
    # if the mask is all empty
    if np.all(np.isclose(mask_data, 0)):
        mask_data = np.ones(mask_data.shape)

    # return the mask
    return mask_data.astype(np.bool8)


# def fix_unwrapping_errors(
#     unwrapped: npt.NDArray[np.float64], mask: npt.NDArray[np.bool8], echos: Union[List[int], None] = None
# ) -> npt.NDArray:
#     # get echoes to try and fix
#     if echos is None:
#         echos = [0, 1]
#     echos = cast(List[int], echos)

#     # generate finite difference filters
#     filters = np.zeros((3, 3, 3, 6))
#     filters[1, 1, 1] = 1
#     n = 0
#     for i in [-1, 1]:
#         filters[1 + i, 1, 1, n] = -1
#         n += 1
#     for j in [-1, 1]:
#         filters[1, 1 + j, 1, n] = -1
#         n += 1
#     for k in [-1, 1]:
#         filters[1, 1, 1 + k, n] = -1
#         n += 1

#     # compute structuring element for dilation
#     strel = generate_binary_structure(3, 1)
#     import simplebrainviewer as sbv

#     # loop over echos
#     for idx in echos:
#         phase3 = unwrapped[..., idx].copy()
#         # get the phase
#         phase = unwrapped[..., idx]

#         # get the integer map
#         int_map = np.round(phase / (2 * np.pi)).astype(int)

#         # now get the area where integer map is 0
#         # we will use this region as the seed for the growing region
#         # and mask with the brain mask
#         visited_region = (int_map == 0) & mask

#         # loop until we visited all voxels in mask
#         while not np.all(visited_region == mask):
#             # dilate the visited region
#             expanded_region = binary_dilation(visited_region, structure=strel, mask=mask)

#             # xor the original region to get next set of voxels
#             next_edge_set = expanded_region ^ visited_region

#             # integer offsets to test
#             offsets = [-1, 0, 1]

#             # get valid voxel masks by using the finite difference filters
#             # and masking out any voxels not in the visited region
#             valid = np.zeros((*phase.shape, filters.shape[-1]))
#             for f in range(filters.shape[-1]):
#                 # find valid voxels using this finite difference filter
#                 # the convolution will return 0 for voxels that are valid
#                 # so we need to take the complement and mask with the edge set
#                 valid[..., f] = ~convolve(expanded_region, filters[..., f], mode="nearest") & next_edge_set

#             # create array to store sum squared errors
#             # last axis is for the integer offsets we are testing
#             errors = np.zeros((*phase.shape, len(offsets)))

#             # for each offset add 2pi * offset to the phase to test
#             # if it is a better solution
#             for i, offset in enumerate(offsets):
#                 # make a copy of the phase
#                 phase_wc = phase.copy()

#                 # add the 2pi offset to working copy (only for edge voxels)
#                 phase_wc[next_edge_set] += 2 * np.pi * offset

#                 # for each filter, compute the squared finite difference and mask with valid
#                 # store in errors for this offset
#                 for f in range(filters.shape[-1]):
#                     errors[..., i] += (convolve(phase_wc, filters[..., f], mode="nearest") ** 2) * valid[..., f]

#             # get the best solution for the set
#             # add the first offset to go from idx to offset value
#             best_solution = np.argmin(errors, axis=-1) + offsets[0]
#             if np.all(best_solution == 0):
#                 break

#             # now add the best solution to the phase
#             phase[next_edge_set] += 2 * np.pi * best_solution[next_edge_set]

#             # update visited region
#             visited_region = expanded_region
#         test = np.stack((phase, phase3), axis=3)
#         breakpoint()


def smooth_discontinuities(
    unwrapped: npt.NDArray[np.float64],
    mag_data: npt.NDArray[np.float64],
    threshold: float = 5,
    echos: Union[List[int], None] = None,
    filter_size: int = 9,
) -> npt.NDArray[np.float64]:
    """Smooth discontinuities in the phase"""
    # get echos to fix
    if echos is None:
        echos = [0, 1]
    echos = cast(List[int], echos)

    # create structuring element for dilation
    strel = ball(filter_size // 2)
    strel[:, :, (-filter_size // 2) + 1 :] = 0
    strel[:, :, : (filter_size // 2)] = 0

    # generate finite difference filters
    filters = np.zeros((3, 3, 3, 4))
    filters[1, 1, 1] = 1
    n = 0
    for i in [-1, 1]:
        filters[1 + i, 1, 1, n] = -1
        n += 1
    for j in [-1, 1]:
        filters[1, 1 + j, 1, n] = -1
        n += 1
    # for k in [-1, 1]:
    #     filters[1, 1, 1 + k, n] = -1
    #     n += 1

    # allocate data for bad voxels and norm constant
    bad_voxel_mask = np.zeros((*unwrapped.shape[:3], len(echos)), dtype=bool)
    norm_constant = np.zeros((*unwrapped.shape[:3], len(echos)))

    # for each echo
    for e in echos:
        # create brain mask
        mask = create_brain_mask(mag_data[..., e])

        # get valid gradient mask by using the finite difference filters
        valid_grad = np.zeros((*unwrapped.shape[:3], filters.shape[-1]))
        for f in range(filters.shape[-1]):
            # find valid gradients using this finite difference filter
            # the convolution will return 0 for voxels that are valid
            # so we need to take the complement
            valid_grad[..., f] = ~convolve(mask, filters[..., f], mode="nearest") & mask

        # convolve filters on phase data for each echo
        for f in range(filters.shape[-1]):
            # get the directional gradient for this filter
            fd_grad = convolve(unwrapped[..., e], filters[..., f], mode="nearest") * valid_grad[..., f]
            # if the gradient is greater than the threshold, mark the voxel as bad
            bad_voxel_mask[..., e] |= np.abs(fd_grad) > threshold

        # dilate the bad voxel mask
        bad_voxel_mask[..., e] = binary_dilation(bad_voxel_mask[..., e], structure=strel, mask=mask)

        # get the convolution over the mask, this will be used to normalize the convolution
        norm_constant[..., e] = convolve(mask.astype("f8"), strel.astype("f8"), mode="nearest")

    # compute convolution of phase data with average filter with mask
    unwrapped_smooth = np.zeros((*unwrapped.shape[:3], len(echos)))
    for e in echos:
        unwrapped_smooth[..., e] = convolve(unwrapped[..., e] * mask, strel.astype("f8"), mode="nearest")
        np.divide(
            unwrapped_smooth[..., e],
            norm_constant[..., e],
            out=unwrapped_smooth[..., e],
            where=norm_constant[..., e] != 0,
        )
    # where the bad voxel mask is true, replace the voxel with the smoothed phase
    for e in echos:
        unwrapped[..., : len(echos)][bad_voxel_mask] = unwrapped_smooth[bad_voxel_mask]

    # return the phase data with discontinuities smoothed
    return unwrapped


def compute_fieldmap(
    phase_data: npt.NDArray[np.float64],
    mag_data: npt.NDArray[np.float64],
    TEs: npt.NDArray[np.float64],
    mask_data: npt.NDArray[np.bool8],
    sigma: npt.NDArray[np.float64],
    automask: bool = True,
    correct_global: bool = True,
    idx: Union[int, None] = None,
) -> npt.NDArray[np.float64]:
    """Computes the field map for a single ME frame.

    Parameters
    ----------
    phase_data : npt.NDArray[np.float64]
        Single frame of phase data with shape (x, y, z, echo)
    mag_data : npt.NDArray[np.float64]
        Single frame of magnitude data with shape (x, y, z, echo)
    TEs : npt.NDArray[np.float64]
        Echo times associated with each phase
    sigma : npt.NDArray[np.float64]
        Sigma for mcpc3ds
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
        field map in Hz
    """
    # use global Julia Context
    global JULIA

    if idx is not None:
        logging.info(f"Processing frame {idx}")

    # if automask is True, generate a mask for the frame, instead of using mask_data
    if automask:
        # get the index with the shortest echo time
        echo_idx = np.argmin(TEs)
        mag_shortest = mag_data[..., echo_idx]

        # create the mask
        # mask_data = JULIA.mri_robustmask(mag_shortest).astype(np.bool8)
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

    # TODO: Remove? Not sure we need this anymore...
    # fix slice wrapping errors in bottom slices
    unwrapped = fix_slice_wrapping_errors(unwrapped, mask_data)

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
        for i in range(1, TEs.shape[0]):
            # get matrix with the masked unwrapped data (weighted by magnitude)
            Y = unwrapped[mask_data, :].T
            # make array with multiple offsets to test
            offsets = list(range(-10, 11))
            # create variables to store best residual and best offset
            best_residual = None
            best_offset = None
            for multiple in offsets:
                # get the ordinate matrix for these echoes
                Y_copy = Y.copy()[: i + 1, :]

                # add offset
                Y_copy[i, :] += 2 * np.pi * multiple

                # fit model and compute residuals
                _, residuals = weighted_regression(X[: i + 1], Y_copy, W[: i + 1])

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

            # update the unwrapped matrix/unwrapped for this echo
            unwrapped[mask_data, i] += 2 * np.pi * best_offset
    nib.Nifti1Image(unwrapped, np.eye(4)).to_filename(f"unwrapped{idx:03d}.nii")

    # broadcast TEs to match shape of data
    TEs_data = TEs[:, np.newaxis]

    # # compute the B0 field from all the phase data
    # B0 = np.zeros(unwrapped.shape[:3])
    # numer = np.sum(unwrapped * (mag_data**2) * TEs_data, axis=-1)
    # denom = np.sum((mag_data**2) * (TEs_data**2), axis=-1)
    # np.divide(numer, denom, out=B0, where=(denom != 0))
    unwrapped_mat = unwrapped.reshape(-1, TEs.shape[0]).T
    weights = mag_data.reshape(-1, TEs.shape[0]).T
    B0 = weighted_regression(TEs_data, unwrapped_mat, weights)[0].T.reshape(*mag_data.shape[:3])
    B0 *= 1000 / (2 * np.pi)

    # save the field map
    return B0


def apply_temporal_consistency(
    unwrapped_data: npt.NDArray, motion_params: npt.NDArray, rd_threshold: float = 0.2
) -> npt.NDArray:
    """Ensures phase unwrapping solutions are temporally consistent

    Parameters
    ----------
    unwrapped_data : npt.NDArray
        unwrapped phase data, where last column is time, and second to last column are the echoes
    motion_params : npt.NDArray
        motion parameters, ordered as x, y, z, rot_x, rot_y, rot_z, note that rotations should be preconverted
        to mm
    rd_threshold : float
        relative displacement threshold. An exponential decay is used to weight each frame, at this threshold
        the weight magnitude is set to 0.05.

    Returns
    -------
    npt.NDArray
        fixed unwrapped phase data
    """
    # make a copy of the unwrapped_data
    fixed_unwrapped_data = unwrapped_data.copy()

    # loop over each frame of the data
    for t in range(unwrapped_data.shape[-1]):
        # now we need to compute the relative displacements to the current frame
        # first get the motion parameters for the current frame
        current_frame_motion_params = motion_params[t][np.newaxis, :]

        # now get the difference of each motion params to this frame
        motion_params_diff = motion_params - current_frame_motion_params

        # now compute the relative displacement by sum of the absolute values
        RD = np.sum(np.abs(motion_params_diff), axis=1)

        # # with the relative displacements computed, figure out the weighting
        # weight_decay_constant = -rd_threshold / np.log(0.05)
        # weights = np.exp(-RD / weight_decay_constant)

        # threhold the RD
        tmask = RD < rd_threshold

        # get the set of frames within the tmask
        frames = fixed_unwrapped_data[..., tmask]

        # for all of the data, compute the 2pi range the data is on via integer multiple
        int_map = np.round(frames / (2 * np.pi)).astype(int)

        # # get the unique set across time
        # unique_set = np.unique(int_map, axis=-1)

        # # compute the weighted mean
        # weighted_mean = np.average(int_map, weights=weights, axis=-1)[..., np.newaxis]

        # # for each voxel find the offset closest to the weighted mean
        # offsets = unique_set[np.argmin(np.abs(unique_set - weighted_mean), axis=-1)]

        # for each voxel find the offset that occurs most frequently
        offsets = mode(int_map, axis=-1).mode

        # for the current frame use the computed offsets
        fixed_unwrapped_data[..., t] += 2 * np.pi * (offsets - int_map)

    # return temporally consistent data
    return fixed_unwrapped_data


def unwrap_and_compute_field_maps(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float], npt.NDArray[np.float64]],
    mask: Union[nib.Nifti1Image, SimpleNamespace, None] = None,
    automask: bool = True,
    correct_global: bool = True,
    frames: Union[List[int], None] = None,
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
        # get the total number of frames
        n_frames = phase[0].shape[3]
        # if frames is None, set it to all frames
        if frames is None:
            frames = np.arange(n_frames)
    else:
        raise ValueError("Data must be 3D or 4D.")
    # frames should be a list at this point
    frames = cast(List[int], frames)

    # check echo times = number of mag and phase images
    if len(TEs) != len(phase) or len(TEs) != len(mag):
        raise ValueError("Number of echo times must equal number of mag and phase images.")

    # allocate space for field maps
    field_maps = np.zeros((*phase[0].shape[:3], n_frames))

    # allocate mask if needed
    if not mask:
        mask = SimpleNamespace()
        mask.dataobj = np.ones((*phase[0].shape[:3], n_frames))

    # get voxel size
    voxel_size = np.array(phase[0].header.get_zooms()[:3])
    sigma = np.array([7, 7, 7]) / voxel_size

    # use a process pool to speed up the computation
    executor = ProcessPoolExecutor(max_workers=n_cpus, initializer=initialize_julia_context)

    # set error flag for process pool
    PROCESS_POOL_ERROR = False

    try:
        # collect futures in a dictionary, where the value is the frame index
        futures = dict()
        # loop over the total number of frames
        for idx in frames:
            # get the phase and magnitude data from each echo
            phase_data: npt.NDArray[np.float64] = rescale_phase(
                np.stack([p.dataobj[..., idx] for p in phase], axis=-1)
            ).astype(np.float64)
            mag_data: npt.NDArray[np.float64] = np.stack([m.dataobj[..., idx] for m in mag], axis=-1).astype(np.float64)
            mask_data = cast(npt.NDArray[np.bool8], mask.dataobj[..., idx].astype(bool))

            # # FOR DEBUGGING
            # initialize_julia_context()
            # field_maps[..., idx] = compute_fieldmap(
            #     phase_data, mag_data, TEs, mask_data, sigma, automask, correct_global, idx
            # )

            # submit field map computation to the process pool
            futures[
                executor.submit(
                    compute_fieldmap,
                    phase_data,
                    mag_data,
                    TEs,
                    mask_data,
                    sigma,
                    automask,
                    correct_global,
                    idx,
                )
            ] = idx

        # loop over the futures
        for future in as_completed(futures):
            # get the frame index
            idx = futures[future]
            # get the field map
            field_maps[..., idx] = future.result()
    except KeyboardInterrupt:
        # on keyboard interrupt, just kill all processes in the executor pool
        for proc in executor._processes.values():
            proc.kill()
        # set global error flag
        PROCESS_POOL_ERROR = True
    except Exception:
        import traceback
        traceback.print_exc()
        # set global error flag
        PROCESS_POOL_ERROR = True
    finally:
        if PROCESS_POOL_ERROR:
            sys.exit(1)
        # close the executor
        else:
            executor.shutdown()
    unwrapped_phases = []
    for i in range(len(frames)):
        unwrapped = nib.load(f"unwrapped{i:03d}.nii")
        nib.Nifti1Image(unwrapped.get_fdata(), phase[0].affine).to_filename(f"unwrapped{i:03d}_af.nii")
        unwrapped_phases.append(nib.load(f"unwrapped{i:03d}.nii"))
    for k in range(5):
        data = np.stack([img.dataobj[..., k] for img in unwrapped_phases], axis=3)
        nib.Nifti1Image(data, phase[0].affine).to_filename(f"unwrapped_phase_echo{k+1}.nii")
    # from memori.pathman import PathManager as PathMan
    # for i in range(len(frames)):
    #     PathMan(f"unwrapped{i:03d}.nii").unlink(missing_ok=True)
    # return the field map as a nifti image
    return nib.Nifti1Image(field_maps[..., frames], phase[0].affine, phase[0].header)


def compute_wrap_error_mask(
    data: npt.NDArray[np.float64],
    brain_mask: npt.NDArray[np.bool8],
    threshold: float = 2 * np.pi,
    area_threshold: int = 100,
):
    """Compute the wrap error mask.

    Tries to find large regions of wrapping errors in the data. This is done by computing a difference across the
    slices, marking any voxels larger than the threshold. The filtering regions not itersecting with the brain mask
    and smaller than the area threshold.

    Parameters
    ----------
    data : npt.NDArray[np.float64]
        The data to compute the wrap error mask for.
    unwrapping_mask : npt.NDArray[np.bool8]
        The mask to use for unwrapping.
    brain_mask : npt.NDArray[np.bool8]
        The brain mask to use for filtering.
    threshold : float
        The threshold to use for the difference.
    area_threshold : int, optional
        The area threshold to use for filtering, by default 100

    Returns
    -------
    npt.NDArray[np.bool8]
        The wrap error mask.
    """

    # compute the slice-to-slice differences
    wrap_errors = (np.abs(np.diff(data, axis=2, append=0)) > threshold) & brain_mask

    # eliminate anything in upper 4/5 of FOV
    wrap_errors[..., data.shape[2] // 5 :] = False

    # now loop through each slice
    for slice_idx in range(data.shape[2]):
        # label the errors of slice
        labelled_slice = label(wrap_errors[..., slice_idx])
        props = regionprops(labelled_slice)
        # for the current slice, region must intersect with brain_mask and have area > area_threshold
        for prop in props:
            # check if current region has area > area_threshold
            if prop.area < area_threshold:
                # delete the region if it does not have area > area_threshold
                wrap_errors[..., slice_idx][prop.label == labelled_slice] = False

    # return the wrap error mask
    return wrap_errors


def estimate_good_slices(wrap_errors: npt.NDArray[np.bool8]) -> Tuple[int, int]:
    """Try to find indices of good slices to correct to.

    A good set of slices is a contiguous set of slices that have no wrapping errors.

    We search from top to bottom for a contiguous set of slices that have no wrapping errors.

    Parameters
    ----------
    wrap_errors : npt.NDArray[np.bool8]
        Mask of wrapping errors

    Returns
    -------
    Tuple[int, int]
        The start and end slice indices of the good slices.
    """
    # get slice indices of longest subvolume where there are no wrapping errors
    good_slices = np.where(~np.any(wrap_errors, axis=(0, 1)))[0]
    if good_slices.size == 0:
        # s^&%t's broken, tell the user their data is bad and just return the middle slice
        print("No good slices found, using middle slice, but slice wrap error correction may fail...")
        return wrap_errors.shape[2] // 2, wrap_errors.shape[2] // 2
    best_slice_set = sorted(np.split(good_slices, np.where(np.diff(good_slices) != 1)[0] + 1), key=lambda x: -len(x))[0]
    return best_slice_set[0], best_slice_set[-1]


def fix_slice_wrapping_errors(
    unwrapped: npt.NDArray[np.float64],
    brain_mask: npt.NDArray[np.bool8],
    thresholds: List[float] = [5, 5, 5, 5, 5],
    num_echoes: int = 5,
    area_threshold: int = 50,
) -> npt.NDArray[np.float64]:
    """Fix slice wrapping errors left over in unwrapped image.

    Attempts to fix wrapping errors left over in unwrapped image in slice. This only tries to fix issues
    in the bottom of the brain.

    Parameters
    ----------
    unwrapped : npt.NDArray[np.float64]
        Unwrapped phase images after unwrapping algorithm
    brain_mask : npt.NDArray[np.bool8]
        Brain mask
    threshold : List[float], optional
        Threshold for wrap error correction, by default [5, 5, 5, 5, 5]
    num_echoes : int, optional
        Number of echoes to try to correct. Later echoes are heavily wrapped and so maybe difficult to correct,
        by default 5
    area_threshold : int, optional
        Minimum size of region to try and correct (this only is relevant for the first initial slice), by default 50

    Returns
    -------
    npt.NDArray[np.float64]
        Fixed Unwrapped phase images
    """
    corrected_echo_data = unwrapped.copy()
    # for each echo
    for echo_idx in range(num_echoes):
        echo_data = unwrapped[..., echo_idx]
        # compute the initial wrap_errors mask
        wrap_errors = compute_wrap_error_mask(echo_data, brain_mask, thresholds[echo_idx], area_threshold)

        # get best contiguous set of good slices
        slice_bottom, _ = estimate_good_slices(wrap_errors)

        # make a working copy of the echo_data
        working_echo_data = echo_data.copy()

        # make a copy of the wrap_errors mask
        working_wrap_errors = (wrap_errors).copy()

        # start from best bottom slice and work down
        for slice_idx in range(slice_bottom, -1, -1):
            while np.any(working_wrap_errors[..., slice_idx]):
                # get wrap errors for current slice
                current_slice_wrap_errors = working_wrap_errors[..., slice_idx]

                # get the current slice and the slice above
                current_slice = working_echo_data[..., slice_idx]
                slice_above = working_echo_data[..., slice_idx + 1]

                # compute the difference between the current slice and the slice above
                diff = slice_above - current_slice
                # find the nearest pi multiple for the difference
                wrap_multiples = np.round(diff / (2 * np.pi))

                # label each error region and loop through each region
                labelled_errors = label(current_slice_wrap_errors)
                props = regionprops(labelled_errors)
                for prop in props:
                    # get the region
                    region = prop.label == labelled_errors

                    # intersect with the brain mask
                    region_with_mask = region & brain_mask[..., slice_idx]

                    # get the set of wrap_multiple values for the region
                    region_wrap_multiples = wrap_multiples[region_with_mask]

                    # compute the mode of the regions wrap multiple values
                    correction_multiple = mode(region_wrap_multiples, keepdims=False).mode

                    # apply the correction to the current slice
                    working_echo_data[region, slice_idx] += correction_multiple * 2 * np.pi

                    # now compute new wrap errors
                    working_wrap_errors = compute_wrap_error_mask(
                        working_echo_data, brain_mask, thresholds[echo_idx], area_threshold=0
                    )

        # save the corrected echo data
        corrected_echo_data[..., echo_idx] = working_echo_data

    # return the corrected echo data
    return corrected_echo_data
