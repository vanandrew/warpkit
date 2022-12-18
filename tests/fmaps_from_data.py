from pathlib import Path
from bids import BIDSLayout
from warpkit.distortion import medic
from warpkit.utilities import displacement_maps_to_field_maps


output_dir = Path("/home/vanandrew/Data/bidsdata_sustest/derivatives/medic")
output_dir.mkdir(exist_ok=True, parents=True)
layout = BIDSLayout("/home/vanandrew/Data/bidsdata_sustest")
runs = layout.get_runs(datatype="func", suffix="bold", extension="nii.gz")

for run in runs:
    print(f"Processing run {run}...")
    mag = layout.get(run=run, part="mag", datatype="func", suffix="bold", extension="nii.gz")
    phase = layout.get(run=run, part="phase", datatype="func", suffix="bold", extension="nii.gz")
    TEs = [m.get_metadata()["EchoTime"] * 1000 for m in mag]
    total_readout_time = phase[0].get_metadata()["TotalReadoutTime"]
    phase_encoding_direction = phase[0].get_metadata()["PhaseEncodingDirection"]
    _, dmaps = medic(
        [i.get_image() for i in phase],
        [i.get_image() for i in mag],
        TEs,
        total_readout_time,
        phase_encoding_direction,
        n_cpus=4,
    )
    dmaps.to_filename(output_dir / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_dmaps.nii.gz")
    # now convert the dmaps back to field maps
    fmaps = displacement_maps_to_field_maps(dmaps, total_readout_time, phase_encoding_direction, True)
    fmaps.to_filename(output_dir / f"sub-MSCHD02_ses-sustest_task-rest_run-{run}_fmaps.nii.gz")
    print(f"Done processing run {run}.")
