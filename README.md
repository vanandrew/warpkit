# mosaic
Multi-echO based Susceptibility ArtIfact Correction (Tentative name, subject to change)

## Installation
To install, clone this repo and run the following in the repo directory:
```
pip install ./ -v
```

> :warning: You will need Julia with the ROMEO package pre-installed on your system.

## Usage

There is currently only a python interface:
```python
import nibabel as nib
from mosaic.unwrap import unwrap_and_compute_field_maps


# paths to your phase and magnitude images...
# phases_paths, magnitude_paths, TEs...

# load phase and magnitude images into lists
# each element in list is a different echo
phases = [nib.load(p) for p in phases_paths]
magnitudes = [nib.load(p) for p in magnitude_paths]

# call the unwrapper and field map computer
field_maps = unwrap_and_compute_field_maps(phases, magnitudes, TEs)

# save the field maps
field_maps.to_filename("/path/to/save.nii.gz")

```
