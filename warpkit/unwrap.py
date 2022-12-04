from types import SimpleNamespace
import signal
from typing import List, Tuple, Union
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

# The Julia init overrides the default python SIGINT handler, so we need to restore it
# after Julia is initialized.
original_sigint_handler = signal.getsignal(signal.SIGINT)

# This should only be initialized once
# so we make it a global that can be used anywhere
JULIA = _JuliaContext()

# Now that Julia is initialized, we can restore the original SIGINT handler
signal.signal(signal.SIGINT, original_sigint_handler)


def unwrap_and_compute_field_maps(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float]],
    mask: Union[nib.Nifti1Image, SimpleNamespace, None] = None,
    automask: bool = True,
    correct_global: bool = True,
    up_to_frame: Union[int, None] = None,
) -> nib.Nifti1Image:
    """Unwrap phase of data weighted by magnitude data and compute field maps.

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
    up_to_frame : int, optional
        Only use up to this frame, by default None

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz
    """
    # check TEs if < 0.1, tell user they probably need to convert to ms
    if np.min(TEs) < 0.1:
        print("WARNING: TEs are unusually small. Your inputs may be incorrect. Did you forget to convert to ms?")

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
        # if up_to_frame is set, make sure it is not greater than n_frames
        if up_to_frame is not None:
            if up_to_frame > n_frames:
                raise ValueError("up_to_frame is greater than the number of frames in the data.")
            n_frames = up_to_frame
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

    # loop over the total number of frames
    for idx in range(n_frames):
        print(f"Processing frame: {idx}")
        # get the phase and magnitude data from each echo
        phase_data = rescale_phase(np.stack([p.dataobj[..., idx] for p in phase], axis=-1))
        mag_data = np.stack([m.dataobj[..., idx] for m in mag], axis=-1)
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
            if np.all(np.isclose(mask_data, 0)):
                mask_data = np.ones(mask_data.shape).astype(bool)
        else:
            mask_data = mask.dataobj[..., idx]
            mask_data = mask_data.astype(bool)

        # unwrap the phase data
        unwrapped = JULIA.romeo_unwrap_individual(
            phase=phase_data,
            TEs=np.array(TEs),
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
        field_maps[..., idx] = B0

    # return the field map as a nifti image
    return nib.Nifti1Image(field_maps, phase[0].affine, phase[0].header)
