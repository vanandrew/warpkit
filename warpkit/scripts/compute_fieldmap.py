import argparse
import json
from typing import cast

import nibabel as nib
import numpy as np

from warpkit import __version__
from warpkit.unwrap import compute_field_maps
from warpkit.utilities import (
    displacement_maps_to_field_maps,
    field_maps_to_displacement_maps,
    invert_displacement_maps,
    setup_logging,
)

from . import epilog

PE_DIRECTIONS = ("i", "j", "k", "i-", "j-", "k-", "x", "y", "z", "x-", "y-", "z-")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute B0 field maps and EPI distortion-correction displacement "
            "maps from previously unwrapped multi-echo phase. The unwrapped "
            "phase and masks are the outputs of `wk-unwrap-phase`. This is "
            "the post-unwrap half of `wk-medic` and writes the same three "
            "NIfTIs: native-space field map (Hz), displacement maps (mm) and "
            "undistorted-space field map (Hz)."
        ),
        epilog=f"{epilog} 04/24/2026",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--magnitude",
        nargs="+",
        required=True,
        help="Magnitude data, one per echo (used as regression weights).",
    )
    parser.add_argument(
        "--unwrapped",
        nargs="+",
        required=True,
        help="Unwrapped phase data, one per echo (output of `wk-unwrap-phase`).",
    )
    parser.add_argument(
        "--masks",
        required=True,
        help="Per-frame masks NIfTI (output of `wk-unwrap-phase`).",
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        help=(
            "BIDS-style JSON sidecar for each echo. EchoTime (s) is read per "
            "file; TotalReadoutTime and PhaseEncodingDirection are taken from "
            "the first. Mutually exclusive with --TEs / "
            "--total-readout-time / --phase-encoding-direction."
        ),
    )
    parser.add_argument(
        "--TEs",
        dest="tes",
        nargs="+",
        type=float,
        help=(
            "Echo times in milliseconds, one per echo (must match "
            "--unwrapped order). Required unless --metadata is given."
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
        "--border-filt",
        nargs=2,
        type=int,
        default=(1, 5),
        metavar=("PASS1", "PASS2"),
        help="SVD components for the two-pass border filter (default: 1 5).",
    )
    parser.add_argument(
        "--svd-filt",
        type=int,
        default=10,
        help="SVD components for global denoising (default: 10).",
    )
    parser.add_argument(
        "-n", "--n-cpus", type=int, default=4, help="Number of CPUs to use."
    )

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

    echo_times: list[float]
    total_readout_time: float
    phase_encoding_direction: str
    if args.metadata:
        metadatas = []
        for j in args.metadata:
            with open(j) as f:
                metadatas.append(json.load(f))
        echo_times = [float(m["EchoTime"]) * 1000 for m in metadatas]
        total_readout_time = float(metadatas[0]["TotalReadoutTime"])
        phase_encoding_direction = str(metadatas[0]["PhaseEncodingDirection"])
    else:
        echo_times = cast(list[float], args.tes)
        total_readout_time = cast(float, args.total_readout_time)
        phase_encoding_direction = cast(str, args.phase_encoding_direction)

    if len(echo_times) != len(args.unwrapped) or len(echo_times) != len(args.magnitude):
        parser.error(
            f"got {len(echo_times)} echo time(s), {len(args.unwrapped)} "
            f"--unwrapped file(s), and {len(args.magnitude)} --magnitude "
            "file(s); all three must match."
        )

    setup_logging()
    print(f"wk-compute-fieldmap: {args}")

    mag_imgs = [cast(nib.Nifti1Image, nib.load(m)) for m in args.magnitude]
    unwrapped_imgs = [cast(nib.Nifti1Image, nib.load(u)) for u in args.unwrapped]
    masks_img = cast(nib.Nifti1Image, nib.load(args.masks))

    fmaps_native = compute_field_maps(
        unwrapped_imgs,
        masks_img,
        mag_imgs,
        echo_times,
        border_filt=tuple(args.border_filt),
        svd_filt=args.svd_filt,
        n_cpus=args.n_cpus,
    )

    # convert native-space field maps to displacement maps in distorted space,
    # invert to get distorted -> undistorted, then re-derive an undistorted-
    # space field map. Mirrors warpkit.distortion.medic.
    inv_displacement_maps = field_maps_to_displacement_maps(
        fmaps_native, total_readout_time, phase_encoding_direction
    )
    dmaps = invert_displacement_maps(inv_displacement_maps, phase_encoding_direction)
    fmaps = displacement_maps_to_field_maps(
        dmaps, total_readout_time, phase_encoding_direction, flip_sign=True
    )

    # sign flip if undistorted-space fmap correlates negatively with native
    if (
        np.corrcoef(
            fmaps.dataobj[..., 0].ravel(),
            fmaps_native.dataobj[..., 0].ravel(),
        )[0, 1]
        < 0
    ):
        fmaps = nib.Nifti1Image(fmaps.get_fdata() * -1, fmaps.affine, fmaps.header)

    print("Saving field maps and displacement maps to file...")
    fmaps_native.to_filename(f"{args.out_prefix}_fieldmaps_native.nii")
    dmaps.to_filename(f"{args.out_prefix}_displacementmaps.nii")
    fmaps.to_filename(f"{args.out_prefix}_fieldmaps.nii")
    print("Done.")
