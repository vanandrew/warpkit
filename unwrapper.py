import tempfile
import nibabel as nib
import numpy as np
import shutil
from bids import BIDSLayout
from omni.pipelines.logging import setup_logging
from omni.interfaces.common import run_process
from memori.pathman import PathManager as PathMan


# setup logging
setup_logging()

# Load the data
layout = BIDSLayout('/home/vanandrew/Data/bidsdata/')

# get the magnitude and phase data separately
mag_imgs = layout.get(subject='MSCHD02', datatype='func', task="rest", part="mag", extension='nii.gz')
phase_imgs = layout.get(subject='MSCHD02', datatype='func', task="rest", part="phase", extension='nii.gz')
assert len(mag_imgs) == len(phase_imgs)

# get number of frames in image
mag_img0 = mag_imgs[0].get_image()
nframes = mag_img0.shape[3]
# nframes = 20

# allocate memory for framewise fieldmap
fieldmap = np.zeros(mag_img0.shape)

# make temp folder
with tempfile.TemporaryDirectory() as tmpdir:
    # get temp path
    tmp_path = PathMan(tmpdir)

    # output dir
    out = "test3"
    PathMan(out).mkdir(exist_ok=True)

    # make names for temp each frame mag and phase images
    mag = (tmp_path / "mag.nii.gz").path
    phase = (tmp_path / "phase.nii.gz").path

    # combine echos for each frame, then run romeo to unwrap and obtain the fieldmap
    for frame_num in range(nframes):
        # setup output path
        output = (PathMan(out) / "temp" / f'frame{frame_num:05d}')
        output.mkdir(exist_ok=True, parents=True)
        output = output.path

        # get magnitude and phase data for this frame
        mag_data = np.stack([m.get_image().dataobj[..., frame_num] for m in mag_imgs], axis=-1)
        phase_data = np.stack([p.get_image().dataobj[..., frame_num] for p in phase_imgs], axis=-1)

        # save data to files
        nib.Nifti1Image(mag_data, mag_img0.affine).to_filename(mag)
        nib.Nifti1Image(phase_data, mag_img0.affine).to_filename(phase)

        # make mask of magnitude image
        magnitude0 = (PathMan(tmp_path) / 'magnitude0.nii.gz').path
        nib.Nifti1Image(mag_data[..., 0], mag_img0.affine).to_filename(magnitude0)

        # run bet and erode 1
        magnitude0_bet = (PathMan(tmp_path) / 'magnitude0_bet.nii.gz').path
        magnitude0_bet_mask = (PathMan(tmp_path) / 'magnitude0_bet_mask.nii.gz').path
        mask = (PathMan(output) / 'mask.nii.gz').path
        #run_process(f"bet {magnitude0} {magnitude0_bet} -m -R -v")
        # run_process(f"fslmaths {magnitude0_bet_mask} -ero {mask}")
        #run_process(f"cp {magnitude0_bet_mask} {mask}")

        # run romeo to unwrap the phase data
        run_process(
            f"romeo -p {phase} -m {mag} "
            f"-t [14.2,38.93,63.66,88.39,113.12] "
            f"-o {output} -B -k robustmask "
            #f"--individual-unwrapping --phase-offset-correction off"
            f"-g --template 1 -v")
        # breakpoint()
        # load in the fieldmap
        b0 = nib.load(PathMan(output) / "B0.nii").get_fdata()
        fieldmap[..., frame_num] = b0

# save the fieldmap
nib.Nifti1Image(fieldmap, mag_img0.affine).to_filename((PathMan(out) / "fieldmap.nii.gz").path)

# # remove temp files
# shutil.rmtree((PathMan(out) / "temp").path)
