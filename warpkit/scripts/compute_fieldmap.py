"""``wk-compute-fieldmap`` — post-unwrap stage of MEDIC: compute B0 field maps
and EPI distortion-correction displacement maps from previously unwrapped
multi-echo phase.

Public surface:

* :func:`compute_fieldmap` — typed Python entry point.
* :func:`main` — argparse CLI shim.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from os import PathLike

import nibabel as nib

from warpkit.unwrap import compute_field_maps
from warpkit.utilities import (
    reconstruct_displacement_and_field_maps,
    setup_logging,
)

from . import epilog
from ._cli import add_n_cpus_arg, add_trt_pe_args, add_version_arg
from ._metadata import ensure_image, ensure_images, resolve_acquisition
from ._outputs import MedicResult, write_medic_outputs

# wk-compute-fieldmap writes and returns the same three NIfTIs as wk-medic.
ComputeFieldmapResult = MedicResult


def compute_fieldmap(
    *,
    unwrapped: Sequence[PathLike[str] | str | nib.Nifti1Image],
    magnitude: Sequence[PathLike[str] | str | nib.Nifti1Image],
    masks: PathLike[str] | str | nib.Nifti1Image,
    out_prefix: PathLike[str] | str,
    tes: Sequence[float] | None = None,
    total_readout_time: float | None = None,
    phase_encoding_direction: str | None = None,
    metadata: Sequence[PathLike[str] | str] | None = None,
    border_filt: Sequence[int] = (1, 5),
    svd_filt: int = 10,
    n_cpus: int = 4,
) -> ComputeFieldmapResult:
    """Compute native-space field map, displacement map, and undistorted-space
    field map from already-unwrapped multi-echo phase + masks.

    ``unwrapped`` and ``masks`` are typically the outputs of
    :func:`warpkit.scripts.unwrap_phase.unwrap_phase`.
    """
    tes_ms, trt, ped = resolve_acquisition(
        metadata=metadata,
        tes=tes,
        total_readout_time=total_readout_time,
        phase_encoding_direction=phase_encoding_direction,
        require_trt_pe=True,
    )
    assert trt is not None and ped is not None

    if len(tes_ms) != len(unwrapped) or len(tes_ms) != len(magnitude):
        raise ValueError(
            f"got {len(tes_ms)} echo time(s), {len(unwrapped)} unwrapped "
            f"file(s), and {len(magnitude)} magnitude file(s); all three "
            "must match."
        )

    mag_imgs = ensure_images(magnitude)
    unwrapped_imgs = ensure_images(unwrapped)
    masks_img = ensure_image(masks)

    border_filt_list = list(border_filt)
    if len(border_filt_list) != 2:
        raise ValueError(
            f"border_filt must have exactly 2 elements; got {len(border_filt_list)}."
        )
    border_filt_tuple: tuple[int, int] = (border_filt_list[0], border_filt_list[1])

    fmaps_native = compute_field_maps(
        unwrapped_imgs,
        masks_img,
        mag_imgs,
        list(tes_ms),
        border_filt=border_filt_tuple,
        svd_filt=svd_filt,
        n_cpus=n_cpus,
    )

    # Convert native-space field maps to displacement maps in distorted space,
    # invert to get distorted -> undistorted, then re-derive an undistorted-
    # space field map. Shared with warpkit.distortion.medic.
    dmaps, fmaps = reconstruct_displacement_and_field_maps(fmaps_native, trt, ped)

    return write_medic_outputs(out_prefix, fmaps_native, dmaps, fmaps)


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
    add_version_arg(parser)
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
    add_trt_pe_args(parser)
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
    add_n_cpus_arg(parser)

    args = parser.parse_args()
    setup_logging()
    print(f"wk-compute-fieldmap: {args}")

    try:
        compute_fieldmap(
            unwrapped=args.unwrapped,
            magnitude=args.magnitude,
            masks=args.masks,
            out_prefix=args.out_prefix,
            tes=args.tes,
            total_readout_time=args.total_readout_time,
            phase_encoding_direction=args.phase_encoding_direction,
            metadata=args.metadata,
            border_filt=tuple(args.border_filt),
            svd_filt=args.svd_filt,
            n_cpus=args.n_cpus,
        )
    except ValueError as e:
        parser.error(str(e))
