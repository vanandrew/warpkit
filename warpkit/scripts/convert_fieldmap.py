"""``wk-convert-fieldmap`` — convert between mm displacement maps/fields and
Hz B0 field maps.

Public surface:

* :func:`convert_fieldmap` — typed Python entry point.
* :func:`main` — argparse CLI shim.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import nibabel as nib

from warpkit import __version__
from warpkit.utilities import (
    AXIS_MAP,
    WARP_ITK_FLIPS,
    displacement_field_to_map,
    displacement_map_to_field,
    displacement_maps_to_field_maps,
    field_maps_to_displacement_maps,
    setup_logging,
)

from . import epilog
from ._warp_io import read_input_frames, write_output

PE_DIRECTIONS = tuple(AXIS_MAP)


@dataclass(frozen=True, slots=True)
class ConvertFieldmapResult:
    output: list[Path]


def convert_fieldmap(
    *,
    input: Sequence[PathLike[str] | str | nib.Nifti1Image],
    output: Sequence[PathLike[str] | str],
    from_type: str,
    to_type: str,
    total_readout_time: float,
    phase_encoding_direction: str,
    from_format: str = "itk",
    to_format: str = "itk",
    flip_sign: bool = False,
    frame: int | None = None,
) -> ConvertFieldmapResult:
    """Convert between mm displacement maps/fields and Hz B0 field maps.

    Exactly one of ``from_type`` / ``to_type`` must be ``"fieldmap"`` (the Hz
    side); the other side is ``"map"`` (1-channel mm) or ``"field"``
    (3-channel mm). Use :func:`convert_warp` for representation/format
    conversions on the mm side.
    """
    if from_type not in ("map", "field", "fieldmap"):
        raise ValueError(
            f"from_type must be 'map', 'field' or 'fieldmap'; got {from_type!r}"
        )
    if to_type not in ("map", "field", "fieldmap"):
        raise ValueError(
            f"to_type must be 'map', 'field' or 'fieldmap'; got {to_type!r}"
        )
    if phase_encoding_direction not in PE_DIRECTIONS:
        raise ValueError(
            f"phase_encoding_direction must be one of {PE_DIRECTIONS}; "
            f"got {phase_encoding_direction!r}"
        )
    if from_format not in WARP_ITK_FLIPS:
        raise ValueError(
            f"from_format must be one of {tuple(WARP_ITK_FLIPS)}; got {from_format!r}"
        )
    if to_format not in WARP_ITK_FLIPS:
        raise ValueError(
            f"to_format must be one of {tuple(WARP_ITK_FLIPS)}; got {to_format!r}"
        )

    if from_type == to_type:
        raise ValueError(
            f"from_type={from_type} and to_type={to_type} are the same; use "
            "convert_warp for representation/format conversions on the mm side."
        )

    crosses_units = (from_type == "fieldmap") != (to_type == "fieldmap")
    if not crosses_units:
        raise ValueError(
            "convert_fieldmap converts between mm (map/field) and Hz "
            "(fieldmap); both --from and --to are on the mm side. Use "
            "convert_warp instead."
        )

    # _warp_io.read_input_frames knows about map/field only; remap "fieldmap"
    # to "map" since on disk a Hz field map is shaped like a 1-channel map.
    from_io = "map" if from_type == "fieldmap" else from_type
    frames = read_input_frames(list(input), from_io)

    if frame is not None:
        if frame < 0 or frame >= len(frames):
            raise ValueError(
                f"frame {frame} is out of range; input has {len(frames)} frame(s)"
            )
        frames = [frames[frame]]

    print(
        f"wk-convert-fieldmap: {len(frames)} frame(s); "
        f"{from_type} -> {to_type} "
        f"(trt={total_readout_time}s, pe={phase_encoding_direction})"
    )

    converted: list[nib.Nifti1Image] = []
    for img in frames:
        if to_type == "fieldmap":
            # mm side -> Hz fieldmap
            if from_type == "field":
                map_img = displacement_field_to_map(
                    img, axis=phase_encoding_direction, format=from_format
                )
            else:
                map_img = img
            converted.append(
                displacement_maps_to_field_maps(
                    map_img,
                    total_readout_time,
                    phase_encoding_direction,
                    flip_sign=flip_sign,
                )
            )
        else:
            # Hz fieldmap -> mm side
            map_img = field_maps_to_displacement_maps(
                img, total_readout_time, phase_encoding_direction
            )
            if to_type == "field":
                converted.append(
                    displacement_map_to_field(
                        map_img,
                        axis=phase_encoding_direction,
                        format=to_format,
                        frame=0,
                    )
                )
            else:
                converted.append(map_img)

    out_writer_type = "field" if to_type == "field" else "map"
    out_paths_resolved = [Path(str(p)).resolve() for p in output]
    write_output(converted, [str(p) for p in out_paths_resolved], out_writer_type)
    print("Done.")
    return ConvertFieldmapResult(output=out_paths_resolved)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert between mm displacement maps/fields and Hz B0 field "
            "maps. The conversion uses the EPI total readout time and the "
            "phase-encoding direction (with sign): "
            "displacement = fieldmap * total_readout_time * voxel_size. "
            "Accepts the same per-frame input model as wk-convert-warp "
            "(1+ files, 3D/4D/5D series) and writes one output volume per "
            "input frame (bundled into a single file if given one --output, "
            "or one per frame if given N)."
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
        help=(
            "Input file(s). 1-channel (3D/4D) for displacement maps (mm) or "
            "B0 field maps (Hz); 4D/5D 3-channel for displacement fields (mm)."
        ),
    )
    parser.add_argument(
        "--output",
        nargs="+",
        required=True,
        help="Output path(s). Pass one for a bundled series, or N for one per frame.",
    )
    parser.add_argument(
        "--from",
        dest="from_type",
        choices=("map", "field", "fieldmap"),
        required=True,
        help=(
            "Input type: 'map' (1-channel mm), 'field' (3-channel mm), or "
            "'fieldmap' (1-channel Hz)."
        ),
    )
    parser.add_argument(
        "--to",
        dest="to_type",
        choices=("map", "field", "fieldmap"),
        required=True,
        help="Output type.",
    )
    parser.add_argument(
        "--total-readout-time",
        type=float,
        help="Total readout time in seconds. Required.",
    )
    parser.add_argument(
        "--phase-encoding-direction",
        choices=PE_DIRECTIONS,
        metavar="DIR",
        help=(
            f"Phase encoding direction with sign (one of: "
            f"{', '.join(PE_DIRECTIONS)}). Required."
        ),
    )
    parser.add_argument(
        "--from-format",
        choices=tuple(WARP_ITK_FLIPS),
        default="itk",
        help="Input field format. Used only when --from=field.",
    )
    parser.add_argument(
        "--to-format",
        choices=tuple(WARP_ITK_FLIPS),
        default="itk",
        help="Output field format. Used only when --to=field.",
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help=(
            "Flip the sign of the resulting fieldmap (only used when "
            "--to=fieldmap). Mirrors the flip-sign branch in "
            "warpkit.distortion.medic."
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Optional: convert a single 0-indexed frame from the input.",
    )

    args = parser.parse_args()
    setup_logging()

    if args.total_readout_time is None or not args.phase_encoding_direction:
        parser.error(
            "--total-readout-time and --phase-encoding-direction are "
            "required for mm <-> Hz conversion."
        )

    try:
        convert_fieldmap(
            input=args.input,
            output=args.output,
            from_type=args.from_type,
            to_type=args.to_type,
            total_readout_time=args.total_readout_time,
            phase_encoding_direction=args.phase_encoding_direction,
            from_format=args.from_format,
            to_format=args.to_format,
            flip_sign=args.flip_sign,
            frame=args.frame,
        )
    except ValueError as e:
        msg = str(e)
        msg = msg.replace("from_type=", "--from=")
        msg = msg.replace("to_type=", "--to=")
        msg = msg.replace("convert_fieldmap converts", "wk-convert-fieldmap converts")
        msg = msg.replace("Use convert_warp", "Use wk-convert-warp")
        msg = msg.replace("frame ", "--frame ", 1) if msg.startswith("frame ") else msg
        parser.error(msg)
