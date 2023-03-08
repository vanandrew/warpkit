import numpy as np
import nibabel as nib
from subprocess import run as subprocess_run
from pathlib import Path
from bids import BIDSLayout


output_dir = Path("/home/vanandrew/Data/bidsdata_sustest/derivatives/topup")
output_dir.mkdir(exist_ok=True, parents=True)
layout = BIDSLayout("/home/vanandrew/Data/bidsdata_sustest")
runs = layout.get_runs(datatype="fmap", direction="AP", suffix="epi", extension="nii.gz")

ped_dict = {
    "j": 1,
    "j-": -1,
}
runs = [run for run in runs]


# define function to grab avg fieldmap image
def avg_field_map(img):
    data = img.get_fdata().mean(axis=3)
    return nib.Nifti1Image(data, img.affine, img.header)


# set reference fieldmap image
ref_PA = layout.get(run="01", direction="PA", datatype="fmap", suffix="epi", extension="nii.gz")[0]
avg_ref_fmap = avg_field_map(ref_PA.get_image())
avg_ref_fmap_path = "ref_fmap.nii.gz"
avg_ref_fmap.to_filename(avg_ref_fmap_path)

for run in runs:
    print(f"Processing run {run}...")
    run_AP = layout.get(run=run, direction="AP", datatype="fmap", suffix="epi", extension="nii.gz")[0]
    run_PA = layout.get(run=run, direction="PA", datatype="fmap", suffix="epi", extension="nii.gz")[0]
    total_readout_time = run_AP.get_metadata()["TotalReadoutTime"]
    ped_AP = run_AP.get_metadata()["PhaseEncodingDirection"]
    ped_PA = run_PA.get_metadata()["PhaseEncodingDirection"]
    img_AP = run_AP.get_image()
    img_PA = run_PA.get_image()
    data_AP = img_AP.get_fdata()
    num_AP = data_AP.shape[3]
    data_PA = img_PA.get_fdata()
    num_PA = data_PA.shape[3]
    data = np.concatenate((data_AP, data_PA), axis=3)
    # nib.Nifti1Image(data, img_AP.affine, img_AP.header).to_filename("temp.nii.gz")

    # with open("acqparams.txt", "w") as f:
    #     for _ in range(num_AP):
    #         f.write(f"0 {ped_dict[ped_AP]} 0 {total_readout_time}\n")
    #     for _ in range(num_PA):
    #         f.write(f"0 {ped_dict[ped_PA]} 0 {total_readout_time}\n")

    topup_output = output_dir / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_topup"
    topup_output.mkdir(exist_ok=True, parents=True)
    out_path = str(topup_output / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_topupout")
    iout_path = str(topup_output / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_iout")
    fout_path = str(topup_output / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_fout")
    dfout_path = str(topup_output / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_dfout")
    # subprocess_run(
    #     [
    #         "topup",
    #         "--imain=temp.nii.gz",
    #         "--datain=acqparams.txt",
    #         "--config=b02b0.cnf",
    #         f"--out={out_path}",
    #         f"--iout={iout_path}",
    #         f"--fout={fout_path}",
    #         f"--dfout={dfout_path}",
    #         "-v"
    #     ],
    #     check=True,
    # )
    # get the average fieldmap image (PA)
    avg_fmap = avg_field_map(img_PA)
    avg_fmap_path = str(topup_output / "avg_fmap.nii.gz")
    avg_fmap.to_filename(avg_fmap_path)
    avg_fmap_refaligned_path = str(topup_output / "avg_fmap_refaligned.nii.gz")
    out_mat = str(topup_output / "avg_fmap_refaligned.mat")
    subprocess_run(
        [
            "flirt",
            "-in",
            f"{avg_fmap_path}",
            "-ref",
            f"{avg_ref_fmap_path}",
            "-out",
            f"{avg_fmap_refaligned_path}",
            "-omat",
            f"{out_mat}",
            "-dof",
            "6",
            "-interp",
            "sinc",
            "-v",
        ]
    )
    # apply the transform to the fieldmap
    fout_refaligned_path = str(topup_output / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_fout_refaligned")
    subprocess_run(
        [
            "flirt",
            "-in",
            f"{fout_path}",
            "-ref",
            f"{avg_ref_fmap_path}",
            "-out",
            f"{fout_refaligned_path}",
            "-init",
            f"{out_mat}",
            "-applyxfm",
            "-interp",
            "sinc",
            "-v",
        ]
    )

    print(f"Done processing run {run}.")

# remove reference fieldmap image
Path(avg_ref_fmap_path).unlink()
Path("acqparams.txt").unlink()
