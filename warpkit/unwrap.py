import sys
import signal
import logging
from types import SimpleNamespace
from typing import List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import (
    generate_binary_structure,
    binary_opening,
    binary_dilation,
)
from .utilities import rescale_phase
from . import JuliaContext as _JuliaContext  # type: ignore


def initialize_julia_context():
    """Initialize a Julia context in the global variable JULIA."""

    # This should only be initialized once
    global JULIA
    JULIA = _JuliaContext()

    # Set the process's signal handler to ignore SIGINT
    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])


def compute_fieldmap(
    phase_data: np.ndarray,
    mag_data: np.ndarray,
    TEs: np.ndarray,
    mask_data: np.ndarray,
    automask: bool = True,
    correct_global: bool = True,
    idx: Union[int, None] = None,
) -> np.ndarray:
    """Computes the field map for a single ME frame.

    Parameters
    ----------
    phase_data : np.ndarray
        Single frame of phase data with shape (x, y, z, echo)
    mag_data : np.ndarray
        Single frame of magnitude data with shape (x, y, z, echo)
    TEs : np.ndarray
        Echo times associated with each phase
    mask_data : np.ndarray
        Mask of voxels to use for unwrapping
    automask : bool, optional
        Automatically compute a mask, by default True
    correct_global : bool, optional
        Corrects global n2π offset, by default True
    idx : int, optional
        Index of the frame being processed for verbosity, by default None

    Returns
    -------
    np.ndarray
        field map in Hz
    """
    if idx is not None:
        logging.info(f"Processing frame {idx}")

    # if automask is True, generate a mask for the frame, instead of using mask_data
    if automask:
        # create structuring element
        strel = generate_binary_structure(3, 2)

        # get the index with the shortest echo time
        echo_idx = np.argmin(TEs)
        mag_shortest = mag_data[..., echo_idx]

        # get the otsu threshold
        threshold = threshold_otsu(mag_shortest)

        # get the mask and open
        mask_data = binary_opening(mag_shortest > threshold, structure=strel, iterations=5)

        # now dilate the mask
        mask_data = binary_dilation(mask_data, structure=strel, iterations=10)

        # since we can't have a completely empty mask, set all zeros to ones
        # if the mask is all empty
        if np.all(np.isclose(mask_data, 0)):
            mask_data = np.ones(mask_data.shape).astype(bool)

    # unwrap the phase data
    global JULIA
    unwrapped = JULIA.romeo_unwrap_individual(
        phase=phase_data,
        TEs=TEs,
        weights="romeo",
        mag=mag_data,
        mask=mask_data,
        correct_global=correct_global,
    )

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
    TEs: Union[List[float], Tuple[float]],
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
    TEs : Tuple[float]
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
    TEs = np.array(TEs)

    # make sure affines/shapes are all correct
    for p1, m1 in zip(phase, mag):
        for p2, m2 in zip(phase, mag):
            if not (
                np.allclose(p1.affine, p2.affine)
                and np.allclose(p1.shape, p2.shape)
                and np.allclose(m1.affine, m2.affine)
                and np.allclose(m1.shape, m2.shape)
                and np.allclose(p1.affine, m1.affine)
                and np.allclose(p1.shape, m1.shape)
                and np.allclose(p2.affine, m2.affine)
                and np.allclose(p2.shape, m2.shape)
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

    # check echo times = number of mag and phase images
    if len(TEs) != len(phase) or len(TEs) != len(mag):
        raise ValueError("Number of echo times must equal number of mag and phase images.")

    # allocate space for field maps
    field_maps = np.zeros((*phase[0].shape[:3], n_frames))

    # allocate mask if needed
    if not mask:
        mask = SimpleNamespace()
        mask.dataobj = np.ones((*phase[0].shape[:3], n_frames))

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
            phase_data = rescale_phase(np.stack([p.dataobj[..., idx] for p in phase], axis=-1))
            mag_data = np.stack([m.dataobj[..., idx] for m in mag], axis=-1)
            mask_data = mask.dataobj[..., idx]
            mask_data = mask_data.astype(bool)

            # submit field map computation to the process pool
            futures[
                executor.submit(compute_fieldmap, phase_data, mag_data, TEs, mask_data, automask, correct_global, idx)
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
    finally:
        if PROCESS_POOL_ERROR:
            sys.exit(1)
        # close the executor
        else:
            executor.shutdown()

    # return the field map as a nifti image
    return nib.Nifti1Image(field_maps, phase[0].affine, phase[0].header)
