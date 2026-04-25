"""``wk-compute-jacobian`` — compute the Jacobian determinant of a
displacement warp.

Public surface:

* :func:`compute_jacobian` — typed Python entry point.
* :func:`main` — argparse CLI shim.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import cast

import nibabel as nib

from warpkit import __version__
from warpkit.utilities import (
    AXIS_MAP,
    WARP_ITK_FLIPS,
    compute_jacobian_determinant,
    convert_warp,
    displacement_map_to_field,
    setup_logging,
)

from . import epilog
from ._warp_io import read_input_frames, write_output


@dataclass(frozen=True, slots=True)
class ComputeJacobianResult:
    output: list[Path]


def _frame_to_itk_field(
    img: nib.Nifti1Image,
    in_type: str,
    in_format: str,
    axis: str | None,
) -> nib.Nifti1Image:
    """Coerce a single map or field frame into an itk-format 3-channel field."""
    if in_type == "map":
        assert axis is not None
        return displacement_map_to_field(img, axis=axis, format="itk", frame=0)
    if in_format == "itk":
        return img
    return cast(nib.Nifti1Image, convert_warp(img, in_type=in_format, out_type="itk"))


def compute_jacobian(
    *,
    input: Sequence[PathLike[str] | str | nib.Nifti1Image],
    output: Sequence[PathLike[str] | str],
    from_type: str,
    from_format: str = "itk",
    axis: str | None = None,
    frame: int | None = None,
) -> ComputeJacobianResult:
    """Compute the per-frame Jacobian determinant of a displacement warp.

    Output is one 3D scalar volume per input frame (bundled into a 4D series
    if a single ``output`` path is given, or one per frame if N paths).
    """
    if from_type not in ("map", "field"):
        raise ValueError(f"from_type must be 'map' or 'field'; got {from_type!r}")
    if from_format not in WARP_ITK_FLIPS:
        raise ValueError(
            f"from_format must be one of {tuple(WARP_ITK_FLIPS)}; got {from_format!r}"
        )
    if axis is not None and axis not in AXIS_MAP:
        raise ValueError(f"axis must be one of {tuple(AXIS_MAP)}; got {axis!r}")

    frames = read_input_frames(list(input), from_type)

    if frame is not None:
        if frame < 0 or frame >= len(frames):
            raise ValueError(
                f"frame {frame} is out of range; input has {len(frames)} frame(s)"
            )
        frames = [frames[frame]]

    if from_type == "map" and not axis:
        raise ValueError("--axis is required when the input is a displacement map.")

    in_fmt_label = from_format if from_type == "field" else "n/a"
    print(
        f"wk-compute-jacobian: {len(frames)} frame(s); {from_type}({in_fmt_label}) "
        "-> jacobian"
    )

    jacobians: list[nib.Nifti1Image] = []
    for img in frames:
        field_itk = _frame_to_itk_field(img, from_type, from_format, axis)
        jacobians.append(compute_jacobian_determinant(field_itk))

    out_paths_resolved = [Path(str(p)).resolve() for p in output]
    write_output(jacobians, [str(p) for p in out_paths_resolved], "map")
    print("Done.")
    return ComputeJacobianResult(output=out_paths_resolved)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Jacobian determinant of a displacement warp. The "
            "input may be 1-channel displacement maps along --axis, or "
            "3-channel displacement fields in any of the ITK / FSL / ANTs / "
            "AFNI conventions; single-frame and multi-frame series are both "
            "accepted with the same input model as wk-convert-warp. The "
            "output is one scalar (Jacobian) volume per input frame: pass "
            "one --output path to bundle into a 4D series, or pass N output "
            "paths for one file per frame. A Jacobian of 1 means no local "
            "volume change; <1 = compression, >1 = expansion."
        ),
        epilog=f"{epilog} 04/25/2026",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input map(s) or field(s). One or more files; any 4D/5D series is split into frames.",
    )
    parser.add_argument(
        "--output",
        nargs="+",
        required=True,
        help=(
            "Output path(s). Pass a single path to bundle Jacobian volumes "
            "into a 4D series, or pass one path per frame."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_type",
        choices=("map", "field"),
        required=True,
        help="Input type: 'map' (1-channel along --axis) or 'field' (3-channel).",
    )
    parser.add_argument(
        "--from-format",
        choices=tuple(WARP_ITK_FLIPS),
        default="itk",
        help="Input field format (itk/fsl/ants/afni). Used only when --from=field.",
    )
    parser.add_argument(
        "--axis",
        choices=tuple(AXIS_MAP),
        metavar="AXIS",
        help=(
            f"Axis the 1-channel maps are along (one of: {', '.join(AXIS_MAP)}). "
            "Required when the input is a displacement map."
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Optional: compute the Jacobian of a single 0-indexed frame from the input.",
    )

    args = parser.parse_args()
    setup_logging()

    try:
        compute_jacobian(
            input=args.input,
            output=args.output,
            from_type=args.from_type,
            from_format=args.from_format,
            axis=args.axis,
            frame=args.frame,
        )
    except ValueError as e:
        msg = str(e)
        msg = msg.replace("frame ", "--frame ", 1) if msg.startswith("frame ") else msg
        parser.error(msg)
