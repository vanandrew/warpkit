import numpy as np
import nibabel as nib


def rescale_phase(data: np.ndarray, min: int = -4096, max: int = 4096) -> np.ndarray:
    """Rescale phase data to [-pi, pi]

    Rescales data to [-pi, pi] using the specified min and max inputs.

    Parameters
    ----------
    data : np.ndarray
        phase data to be rescaled
    min : int, optional
        min value that should be mapped to -pi, by default -4096
    max : int, optional
        max value that should be mapped to pi, by default 4096

    Returns
    -------
    np.ndarray
        rescaled phase data
    """
    return (data - min) / (max - min) * 2 * np.pi - np.pi


def field_maps_to_displacement_maps(
    field_maps: nib.Nifti1Image, effective_echo_spacing: float, phase_encoding_lines: int, voxel_size: float
) -> nib.Nifti1Image:
    """Convert field maps (Hz) to displacement maps (mm)

    Parameters
    ----------
    field_maps : nib.Nifti1Image
        Field map data in Hz
    effective_echo_spacing : float
        Rffective echo spacing in seconds
    phase_encoding_lines : int
        Number of phase encoding lines
    voxel_size : float
        Voxels size in mm

    Returns
    -------
    nib.Nifti1Image
        Displacment maps in mm
    """
    data = field_maps.get_fdata()
    data *= effective_echo_spacing * phase_encoding_lines * voxel_size
    return nib.Nifti1Image(data, field_maps.affine, field_maps.header)
