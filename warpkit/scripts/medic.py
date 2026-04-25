import argparse
import json
from typing import cast

import nibabel as nib

from warpkit import __version__
from warpkit.distortion import medic
from warpkit.utilities import setup_logging

from . import epilog

PE_DIRECTIONS = ("i", "j", "k", "i-", "j-", "k-", "x", "y", "z", "x-", "y-", "z-")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Echo DIstortion Correction", epilog=f"{epilog}"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--magnitude", nargs="+", required=True, help="Magnitude data")
    parser.add_argument("--phase", nargs="+", required=True, help="Phase data")
    parser.add_argument(
        "--metadata",
        nargs="+",
        help=(
            "BIDS-style JSON sidecar for each echo. EchoTime (s) is read per file; "
            "TotalReadoutTime and PhaseEncodingDirection are taken from the first. "
            "Mutually exclusive with --TEs / --total-readout-time / "
            "--phase-encoding-direction."
        ),
    )
    parser.add_argument(
        "--TEs",
        dest="tes",
        nargs="+",
        type=float,
        help=(
            "Echo times in milliseconds, one per echo (must match --phase order). "
            "Required unless --metadata is given."
        ),
    )
    parser.add_argument(
        "--total-readout-time",
        type=float,
        help="Total readout time in seconds. Required unless --metadata is given.",
    )
    parser.add_argument(
        "--phase-encoding-direction",
        choices=PE_DIRECTIONS,
        metavar="DIR",
        help=(
            f"Phase encoding direction (one of: {', '.join(PE_DIRECTIONS)}). "
            "Required unless --metadata is given."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output field maps and displacement maps.",
    )
    parser.add_argument(
        "-f", "--noiseframes", type=int, default=0, help="Number of noise frames"
    )
    parser.add_argument(
        "-n", "--n-cpus", type=int, default=4, help="Number of CPUs to use."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--wrap-limit",
        action="store_true",
        help="Turns off some heuristics for phase unwrapping",
    )

    # parse arguments
    args = parser.parse_args()

    direct_args = {
        "--TEs": args.tes,
        "--total-readout-time": args.total_readout_time,
        "--phase-encoding-direction": args.phase_encoding_direction,
    }
    direct_supplied = [name for name, val in direct_args.items() if val is not None]

    if args.metadata and direct_supplied:
        parser.error(
            "--metadata is mutually exclusive with "
            f"{', '.join(direct_supplied)}; pass one or the other, not both."
        )
    if not args.metadata and len(direct_supplied) != len(direct_args):
        missing = [name for name in direct_args if name not in direct_supplied]
        parser.error(
            "either --metadata or all of --TEs, --total-readout-time, and "
            f"--phase-encoding-direction must be provided (missing: {', '.join(missing)})."
        )

    if args.metadata:
        metadatas = []
        for j in args.metadata:
            with open(j) as f:
                metadatas.append(json.load(f))
        echo_times = [float(m["EchoTime"]) * 1000 for m in metadatas]
        total_readout_time = float(metadatas[0]["TotalReadoutTime"])
        phase_encoding_direction = str(metadatas[0]["PhaseEncodingDirection"])
    else:
        echo_times = args.tes
        total_readout_time = args.total_readout_time
        phase_encoding_direction = args.phase_encoding_direction

    if len(echo_times) != len(args.phase):
        parser.error(
            f"got {len(echo_times)} echo time(s) but --phase has {len(args.phase)} file(s); they must match."
        )

    # setup logging
    setup_logging()

    # log arguments
    print(f"medic: {args}")

    # load magnitude and phase data
    mag_data = [cast(nib.Nifti1Image, nib.load(m)) for m in args.magnitude]
    phase_data = [cast(nib.Nifti1Image, nib.load(p)) for p in args.phase]

    # if noiseframes specified, remove them
    if args.noiseframes > 0:
        print(f"Removing {args.noiseframes} noise frames from the end of each file...")
        mag_data = [
            nib.Nifti1Image(m.dataobj[..., : -args.noiseframes], m.affine, m.header)
            for m in mag_data
        ]
        phase_data = [
            nib.Nifti1Image(p.dataobj[..., : -args.noiseframes], p.affine, p.header)
            for p in phase_data
        ]

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
