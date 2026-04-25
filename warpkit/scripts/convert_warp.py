"""``wk-convert-warp`` — interconvert displacement maps and displacement
fields, convert between ITK / FSL / ANTs / AFNI format conventions, and
optionally invert the warp along the way.

Public surface:

* :func:`convert_warp` — typed Python entry point. Returns a
  :class:`ConvertWarpResult` with the absolute paths of the written NIfTIs.
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
from warpkit.utilities import (
    AXIS_MAP,
    WARP_ITK_FLIPS,
    displacement_field_to_map,
    displacement_map_to_field,
    invert_displacement_field,
    invert_displacement_maps,
    setup_logging,
)
from warpkit.utilities import (
    convert_warp as _convert_warp_image,
)

from . import epilog
from ._warp_io import read_input_frames, write_output


@dataclass(frozen=True, slots=True)
class ConvertWarpResult:
    output: list[Path]


def _invert_frames(
    frames: list[nib.Nifti1Image],
    in_type: str,
    in_format: str,
    axis: str | None,
    verbose: bool,
) -> tuple[list[nib.Nifti1Image], str, str]:
    """Invert each frame and return ``(frames, post_type, post_format)``.

    See module-level commentary in the original script for the routing
    rationale (single-frame vs. multi-frame, map promotion to field, etc.).
    """
    n = len(frames)
    if n == 1:
        img = frames[0]
        if in_type == "map":
            assert axis is not None
            field_itk = displacement_map_to_field(img, axis=axis, format="itk", frame=0)
        else:
            field_itk = (
                img
                if in_format == "itk"
                else _convert_warp_image(img, in_type=in_format, out_type="itk")
            )
        return [invert_displacement_field(field_itk, verbose=verbose)], "field", "itk"

    # N > 1: per-frame -> 1-channel maps -> single batched inversion.
    assert axis is not None
    if in_type == "field":
        map_frames = [
            displacement_field_to_map(f, axis=axis, format=in_format) for f in frames
        ]
    else:
        map_frames = frames
    template = map_frames[0]
    stacked = np.stack([f.get_fdata() for f in map_frames], axis=-1).astype(np.float32)
    stacked_img = nib.Nifti1Image(stacked, template.affine, template.header)
    inverted_4d = invert_displacement_maps(stacked_img, axis=axis, verbose=verbose)
    inv_data = np.asarray(inverted_4d.dataobj)
    inverted = [
        nib.Nifti1Image(inv_data[..., i], inverted_4d.affine, inverted_4d.header)
        for i in range(n)
    ]
    return inverted, "map", in_format


def _convert_frames(
    frames: list[nib.Nifti1Image],
    in_type: str,
    out_type: str,
    in_format: str,
    out_format: str,
    axis: str | None,
) -> list[nib.Nifti1Image]:
    converted: list[nib.Nifti1Image] = []
    for img in frames:
        if in_type == "map" and out_type == "map":
            converted.append(img)
        elif in_type == "field" and out_type == "field":
            if in_format == out_format:
                converted.append(img)
            else:
                converted.append(
                    _convert_warp_image(img, in_type=in_format, out_type=out_format)
                )
        elif in_type == "map" and out_type == "field":
            assert axis is not None
            converted.append(
                displacement_map_to_field(img, axis=axis, format=out_format, frame=0)
            )
        else:  # field -> map
            assert axis is not None
            converted.append(
                displacement_field_to_map(img, axis=axis, format=in_format)
            )
    return converted


def convert_warp(
    *,
    input: Sequence[PathLike[str] | str | nib.Nifti1Image],
    output: Sequence[PathLike[str] | str],
    from_type: str,
    to_type: str | None = None,
    from_format: str = "itk",
    to_format: str = "itk",
    axis: str | None = None,
    frame: int | None = None,
    invert: bool = False,
    verbose: bool = False,
) -> ConvertWarpResult:
    """Convert displacement transforms between map/field representations,
    between field-format conventions, and optionally invert.

    See ``wk-convert-warp --help`` for the full parameter description; the
    Python kwargs are the snake_case equivalents of the CLI flags.
    """
    if from_type not in ("map", "field"):
        raise ValueError(f"from_type must be 'map' or 'field'; got {from_type!r}")
    if to_type is not None and to_type not in ("map", "field"):
        raise ValueError(f"to_type must be 'map' or 'field'; got {to_type!r}")
    if from_format not in WARP_ITK_FLIPS:
        raise ValueError(
            f"from_format must be one of {tuple(WARP_ITK_FLIPS)}; got {from_format!r}"
        )
    if to_format not in WARP_ITK_FLIPS:
        raise ValueError(
            f"to_format must be one of {tuple(WARP_ITK_FLIPS)}; got {to_format!r}"
        )
    if axis is not None and axis not in AXIS_MAP:
        raise ValueError(f"axis must be one of {tuple(AXIS_MAP)}; got {axis!r}")

    out_type = to_type or from_type
    frames = read_input_frames(list(input), from_type)

    if frame is not None:
        if frame < 0 or frame >= len(frames):
            raise ValueError(
                f"frame {frame} is out of range; input has {len(frames)} frame(s)"
            )
        frames = [frames[frame]]

    multi_frame = len(frames) > 1
    needs_axis = (
        (from_type == "map" and out_type == "field")
        or (from_type == "field" and out_type == "map")
        or (invert and (from_type == "map" or multi_frame))
    )
    if needs_axis and not axis:
        raise ValueError(
            "--axis is required when converting between maps and fields, "
            "when inverting a single-frame map, or when inverting a "
            "multi-frame series (the multi-frame inverter operates along a "
            "single axis)."
        )

    in_fmt_label = from_format if from_type == "field" else "n/a"
    out_fmt_label = to_format if out_type == "field" else "n/a"
    invert_label = " (inverted)" if invert else ""
    print(
        f"wk-convert-warp: {len(frames)} frame(s); "
        f"{from_type}({in_fmt_label}) -> {out_type}({out_fmt_label}){invert_label}"
    )

    post_type, post_format = from_type, from_format
    if invert:
        frames, post_type, post_format = _invert_frames(
            frames,
            in_type=from_type,
            in_format=from_format,
            axis=axis,
            verbose=verbose,
        )

    converted = _convert_frames(
        frames,
        in_type=post_type,
        out_type=out_type,
        in_format=post_format,
        out_format=to_format,
        axis=axis,
    )

    out_paths_resolved = [Path(str(p)).resolve() for p in output]
    write_output(converted, [str(p) for p in out_paths_resolved], out_type)
    print("Done.")
    return ConvertWarpResult(output=out_paths_resolved)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interconvert displacement maps and displacement fields, convert "
            "displacement fields between ITK / FSL / ANTs / AFNI format "
            "conventions, and (with --invert) invert the warp along the way. "
            "A single input file may be a 3D map, a 4D map series, a 4D "
            "field, or a 5D field (single or series); multiple input files "
            "are read as a flat series. Outputs are either bundled into one "
            "file (pass one --output path) or written one-per-frame (pass N "
            "output paths). Replaces the older extract_field_from_maps "
            "script."
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
            "Output path(s). Pass a single path to bundle all frames into "
            "one file (4D for maps, 5D for fields), or pass one path per "
            "frame for split output."
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
        "--to",
        dest="to_type",
        choices=("map", "field"),
        help="Output type. Defaults to --from (no type conversion).",
    )
    parser.add_argument(
        "--from-format",
        choices=tuple(WARP_ITK_FLIPS),
        default="itk",
        help="Input field format (itk/fsl/ants/afni). Used only when --from=field.",
    )
    parser.add_argument(
        "--to-format",
        choices=tuple(WARP_ITK_FLIPS),
        default="itk",
        help="Output field format (itk/fsl/ants/afni). Used only when --to=field.",
    )
    parser.add_argument(
        "--axis",
        choices=tuple(AXIS_MAP),
        metavar="AXIS",
        help=(
            f"Axis the 1-channel maps are along (one of: {', '.join(AXIS_MAP)}). "
            "Required when converting between map and field."
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Optional: extract a single 0-indexed frame from the input series.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help=(
            "Invert each frame before any type/format conversion. Routing is "
            "by frame count, not by --from: a single-frame input is inverted "
            "with the full 3D field inverter (a map is first promoted to a "
            "field via --axis), and a multi-frame input is routed through "
            "the per-frame 1D map inverter. For a multi-frame field input, "
            "only the --axis component is inverted and off-axis components "
            "are dropped (fine for EPI distortion correction, where "
            "displacement is along the phase-encoding axis). --axis is "
            "required for any map input and for multi-frame inversion; "
            "single-frame field inversion does not require --axis, but "
            "downstream map output still does."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass-through to the underlying inverter (no effect without --invert).",
    )

    args = parser.parse_args()
    setup_logging()

    try:
        convert_warp(
            input=args.input,
            output=args.output,
            from_type=args.from_type,
            to_type=args.to_type,
            from_format=args.from_format,
            to_format=args.to_format,
            axis=args.axis,
            frame=args.frame,
            invert=args.invert,
            verbose=args.verbose,
        )
    except ValueError as e:
        msg = str(e)
        msg = msg.replace("frame ", "--frame ", 1) if msg.startswith("frame ") else msg
        parser.error(msg)
