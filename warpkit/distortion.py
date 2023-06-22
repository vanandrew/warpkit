from typing import List, Tuple, Optional, Union
import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import signal
from warpkit.unwrap import unwrap_and_compute_field_maps
from warpkit.utilities import (
    field_maps_to_displacement_maps,
    invert_displacement_maps,
    displacement_maps_to_field_maps,
    build_low_pass_filter,
)


def medic(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float]],
    total_readout_time: float,
    phase_encoding_direction: str,
    frames: Optional[List[int]] = None,
    border_size: int = 5,
    border_filt: Tuple[int, int] = (1, 5),
    svd_filt: int = 5,
    critical_freq: Optional[float] = None,
    filter_order: int = 6,
    n_cpus: int = 4,
    debug: bool = False,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image]:
    """This runs Multi-Echo DIstortion Correction (MEDIC) on a set of phase and magnitude images.

    Computes field maps from unwrapped phase images, converts to displacement maps, and inverts to
    get a set of correction maps for a Multi-Echo EPI sequence. Underneath the hood, this uses
    the ROMEO algorithm to unwrap the phase images. To learn more, see the following paper:

    Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K., Robinson, S.D., 2020.
    Phase Unwrapping with a Rapid Opensource Minimum Spanning TreE AlgOrithm (ROMEO).
    Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.28563

    Parameters
    ----------
    phase : List[nib.Nifti1Image]
        Phases to unwrap
    mag : List[nib.Nifti1Image]
        Magnitudes associated with each phase
    TEs : Union[List[float], Tuple[float]]
        Echo times associated with each phase (in miiiseconds)
    total_readout_time : float
        Total readout time (in seconds)
    phase_encoding_direction : str
        Phase encoding direction (can be i, j, k, i-, j-, k-) or (x, y, z, x-, y-, z-)
    frames : int, optional
        Only process these frame indices, by default None (which means all frames)
    border_size : int, optional
        Size of border in automask, by default 5
    border_filt : Tuple[int, int], optional
        Number of SVD components for each step of border filtering, by default (1, 5)
    svd_filt : int, optional
        Number of SVD components to use for filtering of field maps, by default 5
    critical_freq: float, optional
        Critical frequency for low pass filter, by default None, if set to None will disable the filter
    filter_order : int, optional
        Order of the low pass filter, by default 6
    n_cpus : int, optional
        Number of CPUs to use, by default 4
    debug : bool, optional
        Whether to save intermediate files, by default False

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz (distorted space)
    nib.Nifti1Image
        Displacement maps (distorted -> undistorted) in mm
    nib.Nifti1Image
        Field maps in Hz (undistorted space)
    """
    # make sure affines/shapes are all correct
    for p1, m1 in zip(phase, mag):
        for p2, m2 in zip(phase, mag):
            if not (
                np.allclose(p1.affine, p2.affine)
                and np.allclose(m1.affine, m2.affine)
                and p1.shape == p2.shape
                and m1.shape == m2.shape
            ):
                raise ValueError("Affines and shapes must match")

    # unwrap phase and compute field maps
    field_maps_native = unwrap_and_compute_field_maps(
        phase,
        mag,
        TEs,
        border_size=border_size,
        border_filt=border_filt,
        svd_filt=svd_filt,
        frames=frames,
        n_cpus=n_cpus,
        debug=debug,
    )

    # low pass filter field maps if set
    if critical_freq is not None:
        # filter out higher frequencies
        # get the TR from the first mag image
        TR = mag[0].header.get_zooms()[3]
        b, a = build_low_pass_filter(TR, critical_freq, filter_order)
        fmap_data = field_maps_native.get_fdata()
        # apply low pass filter to field maps
        fmap_data = signal.filtfilt(b, a, fmap_data, axis=3)
        field_maps_native = nib.Nifti1Image(fmap_data, field_maps_native.affine, field_maps_native.header)

    # convert to displacement maps (these are in distorted space)
    inv_displacement_maps = field_maps_to_displacement_maps(
        field_maps_native, total_readout_time, phase_encoding_direction
    )

    # invert displacement maps (these are in undistorted space)
    displacement_maps = invert_displacement_maps(inv_displacement_maps, phase_encoding_direction)

    # convert correction maps back to undistorted space field map
    field_maps = displacement_maps_to_field_maps(
        displacement_maps, total_readout_time, phase_encoding_direction, flip_sign=True
    )

    # return correction maps
    return field_maps_native, displacement_maps, field_maps
