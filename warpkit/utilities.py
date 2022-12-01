import numpy as np
import nibabel as nib
from transforms3d.affines import decompose44, decompose
from typing import cast, Union
from .warpkit_cpp import invert_displacement_map


AXIS_MAP = {"x": 0, "y": 1, "z": 2}


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


def displacement_maps_to_field_maps(
    displacement_maps: nib.Nifti1Image, effective_echo_spacing: float, phase_encoding_lines: int, voxel_size: float
) -> nib.Nifti1Image:
    """Convert displacement maps (mm) to field maps (Hz)

    Parameters
    ----------
    displacement_maps : nib.Nifti1Image
        Displacement map data in mm
    effective_echo_spacing : float
        Rffective echo spacing in seconds
    phase_encoding_lines : int
        Number of phase encoding lines
    voxel_size : float
        Voxels size in mm

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz
    """
    data = displacement_maps.get_fdata()
    data /= effective_echo_spacing * phase_encoding_lines * voxel_size
    return nib.Nifti1Image(data, displacement_maps.affine, displacement_maps.header)


def displacement_map_to_field(
    displacement_map: nib.Nifti1Image, axis: str = "y", format: str = "itk"
) -> nib.Nifti1Image:
    """Convert a displacement map to a displacement field

    Parameters
    ----------
    displacement_map : nib.Nifti1Image
        Displacement map data in mm
    axis : str, optional
        Axis displacement maps are along, by default "y"
    format : str, optional
        Format of the displacement field, by default "itk"

    Returns
    -------
    nib.Nifti1Image
        Displacement field in mm
    """
    # get the axis displacement map should insert
    axis_code = AXIS_MAP[axis]
    displacement_map = cast(nib.Nifti1Image, displacement_map)

    # get data, affine and header info
    data = displacement_map.get_fdata()
    affine = displacement_map.affine
    header = cast(nib.Nifti1Header, displacement_map.header)

    # create a new array to hold the displacement field
    new_data = np.zeros((*data.shape, 1, 3))

    # insert data at axis
    new_data[..., axis_code] = data[..., np.newaxis]

    # set the header to the correct intent code
    # this is needed for ANTs, but has no effect on FSL/AFNI
    header.set_intent("vector")

    # change data format based on format
    if format == "itk" or format == "afni":
        # just return the displacement field
        return nib.Nifti1Image(new_data, affine, header)
    elif format == "fsl":
        # for fsl get rid of extra dim
        new_data = new_data.squeeze()

        # multiply y direction by -1
        new_data[..., 1] *= -1

        # return the warp field
        return nib.Nifti1Image(new_data, affine, header)
    else:
        raise ValueError(f"Format {format} not recognized")


def invert_displacement_maps(
    displacement_maps: nib.Nifti1Image, axis: str = "y", verbose: bool = False
) -> nib.Nifti1Image:
    """Invert displacement maps

    Parameters
    ----------
    displacement_maps : nib.Nifti1Image
        Displacement map data in mm
    axis : str, optional
        Axis displacement maps are along, by default "y"
    verbose : bool, optional
        Print debugging information, by default False

    Returns
    -------
    nib.Nifti1Image
        Inverted displacement maps in mm
    """
    # get the axis that this displacement map is along
    axis_code = AXIS_MAP[axis]

    # we convert the data to RAS, for passing into ITK
    # but then return to original coordinate system after
    img_orientation = nib.orientations.io_orientation(displacement_maps.affine)
    ras_orientation = nib.orientations.axcodes2ornt("RAS")
    to_canonical = nib.orientations.ornt_transform(img_orientation, ras_orientation)
    from_canonical = nib.orientations.ornt_transform(ras_orientation, img_orientation)

    # get to RAS orientation
    displacement_maps_ras = displacement_maps.as_reoriented(to_canonical)

    # get data
    data = displacement_maps_ras.dataobj

    # split affine into components
    translations, rotations, zooms, _ = decompose44(displacement_maps_ras.affine)

    # invert maps
    new_data = np.zeros(data.shape)
    for i in range(data.shape[-1]):
        print(f"Processing frame: {i}")
        new_data[..., i] = invert_displacement_map(data[..., i], translations, rotations, zooms, verbose=verbose)

    # make new image in original orientation
    inv_displacement_maps = nib.Nifti1Image(
        new_data, displacement_maps_ras.affine, displacement_maps_ras.header
    ).as_reoriented(from_canonical)

    # return inverted maps
    return cast(nib.Nifti1Image, inv_displacement_maps)
