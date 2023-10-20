import argparse
import json
import nibabel as nib
from . import epilog
from warpkit.distortion import medic
from warpkit.utilities import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Multi-Echo DIstortion Correction", epilog=f"{epilog} 12/09/2022")
    parser.add_argument("--magnitude", nargs="+", required=True, help="Magnitude data")
    parser.add_argument("--phase", nargs="+", required=True, help="Phase data")
    parser.add_argument("--metadata", nargs="+", required=True, help="JSON sidecar for each echo")
    parser.add_argument("--out_prefix", help="Prefix to output field maps and displacment maps.")
    parser.add_argument("-f", "--noiseframes", type=int, default=0, help="Number of noise frames")
    parser.add_argument("-n", "--n_cpus", type=int, default=4, help="Number of CPUs to use.")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--wrap_limit", action="store_true", help="Turns off some heuristics for phase unwrapping")

    # parse arguments
    args = parser.parse_args()

    # setup logging
    setup_logging()

    # log arguments
    print(f"medic: {args}")

    # load magnitude and phase data
    mag_data = [nib.load(m) for m in args.magnitude]
    phase_data = [nib.load(p) for p in args.phase]

    # if noiseframes specified, remove them
    if args.noiseframes > 0:
        print(f"Removing {args.noiseframes} noise frames...")
        mag_data = [nib.Nifti1Image(m.dataobj[..., : -args.noiseframes], m.affine, m.header) for m in mag_data]
        phase_data = [nib.Nifti1Image(p.dataobj[..., : -args.noiseframes], p.affine, p.header) for p in phase_data]

    # get metadata
    echo_times = []
    total_readout_time = None
    phase_encoding_direction = None
    for n, j in enumerate(args.metadata):
        with open(j, "r") as f:
            metadata = json.load(f)
            echo_times.append(metadata["EchoTime"] * 1000)
        if n == 0:
            total_readout_time = metadata["TotalReadoutTime"]
            phase_encoding_direction = metadata["PhaseEncodingDirection"]
    if total_readout_time is None:
        raise ValueError("Could not find TotalReadoutTime in metadata.")
    if phase_encoding_direction is None:
        raise ValueError("Could not find PhaseEncodingDirection in metadata.")

    # now run medic
    if args.debug:
        fmaps_native, dmaps, fmaps = medic(
            phase_data,
            mag_data,
            echo_times,
            total_readout_time,
            phase_encoding_direction,
            n_cpus=args.n_cpus,
            border_filt=(1000, 1000),
            svd_filt=1000,
            debug=True,
            wrap_limit=args.wrap_limit,
        )
    else:
        fmaps_native, dmaps, fmaps = medic(
            phase_data,
            mag_data,
            echo_times,
            total_readout_time,
            phase_encoding_direction,
            n_cpus=args.n_cpus,
            svd_filt=10,
            border_size=5,
            wrap_limit=args.wrap_limit,
        )

    # save the fmaps and dmaps to file
    print("Saving field maps and displacement maps to file...")
    fmaps_native.to_filename(f"{args.out_prefix}_fieldmaps_native.nii")
    dmaps.to_filename(f"{args.out_prefix}_displacementmaps.nii")
    fmaps.to_filename(f"{args.out_prefix}_fieldmaps.nii")
    print("Done.")
