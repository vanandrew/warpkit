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
from scipy.ndimage import (
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
    binary_closing,
    binary_fill_holes,
)
from scipy.stats import mode
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

    # correct phase with mcpc3ds
    # corrected_phase_data = JULIA.mri_mcpc3ds(phase_data, mag_data, TEs, sigma)

    # if automask is True, generate a mask for the frame, instead of using mask_data
    if automask:
        # get the index with the shortest echo time
        echo_idx = np.argmin(TEs)
        mag_shortest = mag_data[..., echo_idx]

        # create the mask
        mask_data = JULIA.mri_robustmask(mag_shortest)
        mask_data = mask_data.astype(np.bool8)

    # unwrap the phase data
    unwrapped = JULIA.romeo_unwrap_individual(
        phase=phase_data,
        TEs=TEs,
        weights="romeo3",
        mag=mag_data,
        mask=mask_data,
        correct_global=False,
        maxseeds=1,
        merge_regions=False,
        correct_regions=False,
    )
    brain_mask = create_brain_mask(mag_data[..., 0])
    unwrapped = fix_slice_wrapping_errors(unwrapped, brain_mask)
    # global median correction
    unwrapped -= np.median(np.round(unwrapped[mask_data, :] / (2 * np.pi)).astype(int), axis=0) * 2 * np.pi
    # global mode correction
    unwrapped -= (
        mode(np.round(unwrapped[mask_data, :] / (2 * np.pi)).astype(int), axis=0, keepdims=False).mode * 2 * np.pi
    )
    # nib.Nifti1Image(unwrapped, np.eye(4)).to_filename(f"unwrapped{idx:03d}.nii")

    # broadcast TEs to match shape of data
    TEs_data = np.array(TEs)[np.newaxis, np.newaxis, np.newaxis, :]

    # compute the B0 field from the phase data
    B0 = np.zeros(unwrapped.shape[:3])
    numer = np.sum(unwrapped * (mag_data**2) * TEs_data, axis=-1)
    denom = np.sum((mag_data**2) * (TEs_data**2), axis=-1)
    np.divide(numer, denom, out=B0, where=(denom != 0))
    B0 *= 1000 / (2 * np.pi)

    # save the field map
    return B0


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
    finally:
        if PROCESS_POOL_ERROR:
            sys.exit(1)
        # close the executor
        else:
            executor.shutdown()
    # unwrapped_phases = []
    # for i in range(len(frames)):
    #     unwrapped = nib.load(f"unwrapped{i:03d}.nii")
    #     nib.Nifti1Image(unwrapped.get_fdata(), phase[0].affine).to_filename(f"unwrapped{i:03d}_af.nii")
    #     unwrapped_phases.append(nib.load(f"unwrapped{i:03d}.nii"))
    # for k in range(5):
    #     data = np.stack([img.dataobj[..., k] for img in unwrapped_phases], axis=3)
    #     nib.Nifti1Image(data, phase[0].affine).to_filename(f"unwrapped_phase_echo{k+1}.nii")
    # from memori.pathman import PathManager as PathMan
    # for i in range(len(frames)):
    #     PathMan(f"unwrapped{i:03d}.nii").unlink(missing_ok=True)
    # return the field map as a nifti image
    return nib.Nifti1Image(field_maps[..., frames], phase[0].affine, phase[0].header)


def compute_wrap_error_mask(
    data: npt.NDArray[np.float64],
    brain_mask: npt.NDArray[np.bool8],
    threshold: float = 2 * np.pi,
    area_threshold: int = 200,
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
        The area threshold to use for filtering, by default 9

    Returns
    -------
    npt.NDArray[np.bool8]
        The wrap error mask.
    """

    # compute the slice-to-slice differences
    wrap_errors = (np.abs(np.diff(data, axis=2, append=0)) > threshold) & brain_mask

    # eliminate anything in upper 3/4 of brain
    wrap_errors[..., data.shape[2] // 4 :] = False

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
    thresholds: List[float] = [3, 3, 6, 6, 6],
    num_echoes: int = 5,
) -> npt.NDArray[np.float64]:
    """Fix slice wrapping errors left over in unwrapped image.

    Attempts to fix wrapping errors left over in unwrapped image in slice. This only tries to fix issues
    in the bottom of the brain.

    Parameters
    ----------
    unwrapped : npt.NDArray[np.float64]
        Unwrapped phase images after unwrapping algorithm
    unwrapping_mask : npt.NDArray[np.bool8]
        Mask used for phase unwrapping
    brain_mask : npt.NDArray[np.bool8]
        Brain mask
    threshold : List[float], optional
        Threshold for wrap error correction, by default [3 * np.pi, 3 * np.pi, 5 * np.pi, 5 * np.pi, 5 * np.pi]
    num_echoes : int, optional
        Number of echoes to try to correct. Later echoes are heavily wrapped and so maybe difficult to correct,
        by default 5

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
        wrap_errors = compute_wrap_error_mask(echo_data, brain_mask, thresholds[echo_idx])

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
                wrap_multiples = np.round(diff / np.pi)

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
                    working_echo_data[region, slice_idx] += correction_multiple * np.pi

                    # now compute new wrap errors
                    working_wrap_errors = compute_wrap_error_mask(
                        working_echo_data, brain_mask, thresholds[echo_idx], area_threshold=1
                    )

        # save the corrected echo data
        corrected_echo_data[..., echo_idx] = working_echo_data

    # return the corrected echo data
    return corrected_echo_data
