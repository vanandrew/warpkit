# warpkit
A python library for neuroimaging transforms

If you've landed here, you're probably interested in the Multi-Echo DIstortion Correction (MEDIC) algorithm, which this
library implements.

See below for usage details.

## Installation
To install, clone this repo and run the following in the repo directory:
```
pip install ./ -v
```
> :warning: You need Julia pre-installed on your system. The installer will raise an error if it is not found.

## MEDIC Usage
The `warpkit` library is meant for integration into a larger python pipeline/package. If you're looking for a script to
run MEDIC, see the [omni](https://gitlab.com/vanandrew/omni) library.

An example of how to call MEDIC from python is provided below:
```python
import nibabel as nib
from warpkit.distortion import medic
from warpkit.utilities import displacement_map_to_warp

# load phase and magnitude images into lists
# each element in list is a different echo
phases = [nib.load(p) for p in phases_paths]
magnitudes = [nib.load(p) for p in magnitude_paths]
TEs = [TE1, TE2, ...] # in milliseconds
effective_echo_spacing = ... # in seconds
phase_encoding_direction = either i, j, k, i-, j-, k-, x , y, z, x-, y-, z- 

# call the medic function
field_maps_native, displacement_maps, field_maps = medic(
    phases, magnitudes, TEs, effective_echo_spacing, phase_encoding_direction)

# field_maps_native are returned in the distorted space (Hz) (mainly for reference/debugging purposes)
# you shouldn't need to use these probably???
# displacement_maps are returned in the undistorted space (mm) (see below for usage)
# field_maps are returned in the undistorted space (Hz) (same field map output as topup/fugue, but framewise)

# returns are nibabel Nifti1Image objects, so they can be saved to file by:

# displacement_maps to file
displacement_maps.to_filename("/path/to/save.nii.gz")

# these should be converted to displacement fields
# by the displacement_map_to_field function
# and specifying the appropriate type (i.e. itk, ants, afni, fsl)

displacement_field = displacement_map_to_field(displacement_maps, axis="y", format="itk", frame=0)

# where axis specifies the phase encoding direction, format is the desired output format, and frame is the index of
# displacement map to convert to a displacement field

# the displacement field can then be saved to file by the to_filename method
# Each file can be applied with the respective software's displacement field application tool:
# itk: the internal format warpkit uses. See utilities.resample_image
# ants: antsApplyTransforms (note that even though ants also uses itk, warpkit's itk warp format is NOT equivalent)
# afni: 3dNwarpApply
# fsl: applywarp

# if you are using fsl and instead want to use fugue to distortion correction, you can use the field_maps outputs
# (these are the equivalent field maps of that you would get from fugue, but with multiple frames)
```
