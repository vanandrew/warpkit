from types import SimpleNamespace
from typing import List, Tuple, Union
import nibabel as nib
import numpy as np

try:
    from mosaic.mosaic_cpp import unwrap  # type: ignore
except ImportError:
    from build.mosaic_cpp import JuliaContext


def unwrap_phases(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float]],
    mask: Union[nib.Nifti1Image, SimpleNamespace, None] = None,
    correctglobal: bool = True,
):
    """Unwrap phase of data weighted by magnitude data.

    Parameters
    ----------
    phase : List[nib.Nifti1Image]
        phases to unwrap
    mag : List[nib.Nifti1Image]
        magnitudes associated with each phase
    TEs : Tuple[float]
        Echo times associated with each phase
    mask : nib.Nifti1Image, optional
        Boolean mask, by default None
    correctglobal : bool, optional
        Corrects global n2π offsets, by default True
    """
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
    else:
        raise ValueError("Data must be 3D or 4D.")

    # check echo times = number of mag and phase images
    if len(TEs) != len(phase) or len(TEs) != len(mag):
        raise ValueError("Number of echo times must equal number of mag and phase images.")

    # allocate space for unwrapped data
    unwrapped = np.zeros(phase[0].shape)

    # allocate mask if needed
    if not mask:
        mask = SimpleNamespace()
        mask.dataobj = np.ones(phase[0].shape, dtype=bool)

    # initalize Julia context
    julia_context = JuliaContext()

    # loop over the total number of frames
    n_frames = 1
    for idx in range(n_frames):
        # get the phase and magnitude data from each echo
        phase_data = np.stack([p.dataobj[..., idx] for p in phase], axis=-1)
        mag_data = np.stack([m.dataobj[..., idx] for m in mag], axis=-1)
        mask_data = mask.dataobj[..., idx]

        julia_context.romeo_unwrap_individual(
            phase=phase_data,
            TEs=TEs,
            weights="romeo",
            mag=mag_data,
            mask=mask_data,
            correctglobal=correctglobal,
        )
