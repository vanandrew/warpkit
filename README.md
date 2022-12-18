# warpkit
A python library for neuroimaging transforms

It includes our implementation of the Multi-Echo DIstortion Correction (MEDIC) algorithm.

## Installation
To install, clone this repo and run the following in the repo directory:
```
pip install ./ -v
```
> :warning: You need Julia pre-installed on your system. The installer will raise an error if it is not found.

## MEDIC Usage

There is currently only a python interface:
```python
import nibabel as nib
from warpkit.distortion import medic

# load phase and magnitude images into lists
# each element in list is a different echo
phases = [nib.load(p) for p in phases_paths]
magnitudes = [nib.load(p) for p in magnitude_paths]
TEs = [TE1, TE2, ...] # in milliseconds
effective_echo_spacing = ... # in seconds
phase_encoding_direction = either i, j, k, i-, j-, k-, x , y, z, x-, y-, z- 

# call the medic function
field_maps, displacement_maps = medic(phases, magnitudes, TEs, effective_echo_spacing, phase_encoding_direction)

# field_maps are returned in the distorted space (Hz)
# displacement_maps are returned in the undistorted space (mm)

# displacement_maps to file
displacement_maps.to_filename("/path/to/save.nii.gz")
# these should be converted to displacement fields
# by extracting a frame and calling the displacement_map_to_warp function
# and specifying the appropriate type (i.e. itk, ants, afni, fsl)
```
