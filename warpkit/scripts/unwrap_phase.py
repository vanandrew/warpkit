import argparse
import json
from typing import cast

import nibabel as nib

from warpkit import __version__
from warpkit.unwrap import unwrap_phases
from warpkit.utilities import setup_logging

from . import epilog


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ROMEO multi-echo phase unwrapping (the unwrap stage of MEDIC). "
            "Outputs one unwrapped phase NIfTI per echo plus the per-frame "
            "automask. Pair with `wk-compute-fieldmap` to obtain a "
            "native-space B0 field map."
        ),
        epilog=f"{epilog} 04/24/2026",
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
            "BIDS-style JSON sidecar for each echo. EchoTime (s) is read per "
            "file. Mutually exclusive with --TEs."
        ),
    )
    parser.add_argument(
        "--TEs",
        dest="tes",
        nargs="+",
        type=float,
        help=(
            "Echo times in milliseconds, one per echo (must match --phase "
            "order). Required unless --metadata is given."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output unwrapped phase and mask files.",
    )
    parser.add_argument(
        "-f", "--noiseframes", type=int, default=0, help="Number of noise frames"
    )
    parser.add_argument(
        "-n", "--n-cpus", type=int, default=4, help="Number of CPUs to use."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Skip the temporal consistency pass and dump intermediate files.",
    )
    parser.add_argument(
        "--wrap-limit",
        action="store_true",
        help="Turn off some heuristics for phase unwrapping.",
    )

    args = parser.parse_args()

    if args.metadata and args.tes is not None:
        parser.error(
            "--metadata is mutually exclusive with --TEs; pass one or the "
            "other, not both."
        )
    if not args.metadata and args.tes is None:
        parser.error("either --metadata or --TEs must be provided.")

    if len(args.magnitude) != len(args.phase):
        parser.error(
            f"got {len(args.magnitude)} --magnitude file(s) but "
            f"{len(args.phase)} --phase file(s); they must match (one "
            "mag/phase pair per echo)."
        )
    if args.metadata is not None and len(args.metadata) != len(args.phase):
        parser.error(
            f"got {len(args.metadata)} --metadata file(s) but "
            f"{len(args.phase)} --phase file(s); they must match (one "
            "sidecar per echo)."
        )

    echo_times: list[float]
    if args.metadata:
        metadatas = []
        for j in args.metadata:
            with open(j) as f:
                metadatas.append(json.load(f))
        echo_times = [float(m["EchoTime"]) * 1000 for m in metadatas]
    else:
        echo_times = cast(list[float], args.tes)

    if len(echo_times) != len(args.phase):
        parser.error(
            f"got {len(echo_times)} echo time(s) but --phase has "
            f"{len(args.phase)} file(s); they must match."
        )

    setup_logging()
    print(f"wk-unwrap-phase: {args}")

    mag_data = [cast(nib.Nifti1Image, nib.load(m)) for m in args.magnitude]
    phase_data = [cast(nib.Nifti1Image, nib.load(p)) for p in args.phase]

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

    unwrapped_imgs, masks_img = unwrap_phases(
        phase_data,
        mag_data,
        echo_times,
        n_cpus=args.n_cpus,
        debug=args.debug,
        wrap_limit=args.wrap_limit,
    )

    print("Saving unwrapped phase images and masks to file...")
    for i_echo, img in enumerate(unwrapped_imgs, start=1):
        img.to_filename(f"{args.out_prefix}_unwrapped_echo-{i_echo:02d}.nii")
    masks_img.to_filename(f"{args.out_prefix}_masks.nii")
    print("Done.")
