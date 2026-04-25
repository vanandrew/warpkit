import argparse

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


def _resolve_in_type(
    from_arg: str,
    shape_type: str,
    to_type: str,
    parser: argparse.ArgumentParser,
) -> str:
    """Disambiguate the input type given ``--from``, the shape-based
    classification (``"map"`` for 1-channel, ``"field"`` for 3-channel),
    and ``--to``."""
    if from_arg != "auto":
        if from_arg == "field" and shape_type != "field":
            parser.error(
                "--from=field expects a 3-channel input but got 1-channel "
                "data; pass --from map or --from fieldmap instead."
            )
        if from_arg in ("map", "fieldmap") and shape_type != "map":
            parser.error(
                f"--from={from_arg} expects a 1-channel input but got "
                "3-channel field data; pass --from field instead."
            )
        return from_arg
    if shape_type == "field":
        return "field"
    # 1-channel input: 'map' (mm) and 'fieldmap' (Hz) look the same on
    # disk, so disambiguate by which side --to is on.
    return "map" if to_type == "fieldmap" else "fieldmap"


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
        choices=("auto", "map", "field", "fieldmap"),
        default="auto",
        help=(
            "Input type. 'auto' uses the shape (3-channel = field) plus "
            "--to to disambiguate map (mm) from fieldmap (Hz) for "
            "1-channel inputs."
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

    # The shape-based classifier in _warp_io knows about map/field only;
    # remap "fieldmap" to "map" for the splitter's purposes.
    from_arg_for_io = "map" if args.from_type == "fieldmap" else args.from_type
    frames, shape_type = read_input_frames(args.input, from_arg_for_io, parser)

    if args.frame is not None:
        if args.frame < 0 or args.frame >= len(frames):
            parser.error(
                f"--frame {args.frame} is out of range; input has "
                f"{len(frames)} frame(s)"
            )
        frames = [frames[args.frame]]

    in_type = _resolve_in_type(args.from_type, shape_type, args.to_type, parser)

    if in_type == args.to_type:
        parser.error(
            f"--from={in_type} and --to={args.to_type} are the same; use "
            "wk-convert-warp for representation/format conversions on the "
            "mm side."
        )

    crosses_units = (in_type == "fieldmap") != (args.to_type == "fieldmap")
    if not crosses_units:
        parser.error(
            "wk-convert-fieldmap converts between mm (map/field) and Hz "
            "(fieldmap); both --from and --to are on the mm side. Use "
            "wk-convert-warp instead."
        )

    if args.total_readout_time is None or not args.phase_encoding_direction:
        parser.error(
            "--total-readout-time and --phase-encoding-direction are "
            "required for mm <-> Hz conversion."
        )

    print(
        f"wk-convert-fieldmap: {len(frames)} frame(s); "
        f"{in_type} -> {args.to_type} "
        f"(trt={args.total_readout_time}s, pe={args.phase_encoding_direction})"
    )

    converted: list[nib.Nifti1Image] = []
    for img in frames:
        if args.to_type == "fieldmap":
            # mm side -> Hz fieldmap
            if in_type == "field":
                map_img = displacement_field_to_map(
                    img,
                    axis=args.phase_encoding_direction,
                    format=args.from_format,
                )
            else:
                map_img = img
            converted.append(
                displacement_maps_to_field_maps(
                    map_img,
                    args.total_readout_time,
                    args.phase_encoding_direction,
                    flip_sign=args.flip_sign,
                )
            )
        else:
            # Hz fieldmap -> mm side
            map_img = field_maps_to_displacement_maps(
                img,
                args.total_readout_time,
                args.phase_encoding_direction,
            )
            if args.to_type == "field":
                converted.append(
                    displacement_map_to_field(
                        map_img,
                        axis=args.phase_encoding_direction,
                        format=args.to_format,
                        frame=0,
                    )
                )
            else:
                converted.append(map_img)

    out_writer_type = "field" if args.to_type == "field" else "map"
    write_output(converted, args.output, out_writer_type, parser)
    print("Done.")
