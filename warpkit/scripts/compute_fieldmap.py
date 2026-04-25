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
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

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
from ._metadata import ensure_image, ensure_images, resolve_acquisition

PE_DIRECTIONS = ("i", "j", "k", "i-", "j-", "k-", "x", "y", "z", "x-", "y-", "z-")


@dataclass(frozen=True, slots=True)
class ComputeFieldmapResult:
    """Same three-tuple as :class:`MedicResult`."""

    fieldmap_native: Path
    displacement_map: Path
    fieldmap: Path


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
    # space field map. Mirrors warpkit.distortion.medic.
    inv_displacement_maps = field_maps_to_displacement_maps(fmaps_native, trt, ped)
    dmaps = invert_displacement_maps(inv_displacement_maps, ped)
    fmaps = displacement_maps_to_field_maps(dmaps, trt, ped, flip_sign=True)

    if (
        np.corrcoef(
            fmaps.dataobj[..., 0].ravel(),
            fmaps_native.dataobj[..., 0].ravel(),
        )[0, 1]
        < 0
    ):
        fmaps = nib.Nifti1Image(fmaps.get_fdata() * -1, fmaps.affine, fmaps.header)

    out_prefix_str = str(out_prefix)
    print("Saving field maps and displacement maps to file...")
    fmap_native_path = Path(f"{out_prefix_str}_fieldmaps_native.nii").resolve()
    dmap_path = Path(f"{out_prefix_str}_displacementmaps.nii").resolve()
    fmap_path = Path(f"{out_prefix_str}_fieldmaps.nii").resolve()
    fmaps_native.to_filename(str(fmap_native_path))
    dmaps.to_filename(str(dmap_path))
    fmaps.to_filename(str(fmap_path))
    print("Done.")
    return ComputeFieldmapResult(
        fieldmap_native=fmap_native_path,
        displacement_map=dmap_path,
        fieldmap=fmap_path,
    )


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
