from tempfile import TemporaryDirectory
from pathlib import Path
import nibabel as nib
from subprocess import run as subprocess_run
from bids import BIDSLayout
from warpkit.distortion import me_sdc


output_dir = Path("/mnt/data/bidsdata_sustest/derivatives/warpkit_mag_align")
output_dir.mkdir(exist_ok=True, parents=True)
layout = BIDSLayout("/mnt/data/bidsdata_sustest")
runs = layout.get_runs(datatype="func", suffix="bold", extension="nii.gz")

with TemporaryDirectory() as tmpdir:
    for idx, run in enumerate(runs):
        mag = layout.get(run=run, part="mag", echo="1", datatype="func", suffix="bold", extension="nii.gz")[0]
        img = mag.get_image()
        data = img.get_fdata()
        if idx == 0:
            pass
        else:
            pass
