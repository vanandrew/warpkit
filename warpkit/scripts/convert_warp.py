import argparse

import nibabel as nib
import numpy as np

from warpkit import __version__
from warpkit.utilities import (
    AXIS_MAP,
    WARP_ITK_FLIPS,
    convert_warp,
    displacement_field_to_map,
    displacement_map_to_field,
    invert_displacement_field,
    invert_displacement_maps,
    setup_logging,
)

from . import epilog
from ._warp_io import read_input_frames, write_output


def _invert_frames(
    frames: list[nib.Nifti1Image],
    in_type: str,
    in_format: str,
    axis: str | None,
    verbose: bool,
) -> tuple[list[nib.Nifti1Image], str, str]:
    """Invert each frame and return ``(frames, post_type, post_format)``.

    Routing is by frame count, not by input type, because the 1D map inverter
    is markedly faster per frame than the full 3D field inverter:

    * **Single frame** uses :func:`invert_displacement_field`. A map input is
      first promoted to a 3-channel itk field via
      :func:`displacement_map_to_field`. The result is always in itk field
      form.
    * **Multi-frame** stacks all frames into a single 4D ``(X, Y, Z, T)`` map
      and runs :func:`invert_displacement_maps` once. A field input first has
      its ``axis`` channel extracted (off-axis channels are dropped — fine
      for the EPI-distortion case, where displacement is along the
      phase-encoding axis). The result is always in 1-channel map form.

    The returned ``post_type`` / ``post_format`` are the actual representation
    of the inverted frames (which may differ from the input's), so the
    downstream conversion stage can route correctly.
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
                else convert_warp(img, in_type=in_format, out_type="itk")
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
                    convert_warp(img, in_type=in_format, out_type=out_format)
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
        choices=("auto", "map", "field"),
        default="auto",
        help="Input type. Default 'auto' uses the NIfTI intent code and shape.",
    )
    parser.add_argument(
        "--to",
        dest="to_type",
        choices=("map", "field"),
        help="Output type. Defaults to whatever the input classifies as.",
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
            "Invert each frame before any type/format conversion. Maps are "
            "inverted with the 1D map inverter along --axis; fields are "
            "inverted with the full 3D field inverter. Inversion of maps "
            "requires --axis (the map's own axis); inversion of fields does "
            "not, but downstream map output still does."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass-through to the underlying inverter (no effect without --invert).",
    )

    args = parser.parse_args()
    setup_logging()

    frames, in_type = read_input_frames(args.input, args.from_type, parser)
    out_type = args.to_type or in_type

    if args.frame is not None:
        if args.frame < 0 or args.frame >= len(frames):
            parser.error(
                f"--frame {args.frame} is out of range; input has "
                f"{len(frames)} frame(s)"
            )
        frames = [frames[args.frame]]

    # --axis is required for map<->field conversion AND for inversion when
    # the chosen inversion routing needs it: the single-frame route promotes
    # a map to a field (needs axis), and the multi-frame route always runs
    # the 1D map inverter (needs axis for map<->axis-channel).
    multi_frame = len(frames) > 1
    needs_axis = (
        (in_type == "map" and out_type == "field")
        or (in_type == "field" and out_type == "map")
        or (args.invert and (in_type == "map" or multi_frame))
    )
    if needs_axis and not args.axis:
        parser.error(
            "--axis is required when converting between maps and fields, "
            "when inverting a single-frame map, or when inverting a "
            "multi-frame series (the multi-frame inverter operates along a "
            "single axis)."
        )

    in_fmt_label = args.from_format if in_type == "field" else "n/a"
    out_fmt_label = args.to_format if out_type == "field" else "n/a"
    invert_label = " (inverted)" if args.invert else ""
    print(
        f"wk-convert-warp: {len(frames)} frame(s); "
        f"{in_type}({in_fmt_label}) -> {out_type}({out_fmt_label}){invert_label}"
    )

    # post_type / post_format track the actual representation of the frames
    # after the inversion stage, which may differ from the user-declared
    # input (e.g. multi-frame field input emerges as 1-channel maps).
    post_type, post_format = in_type, args.from_format
    if args.invert:
        frames, post_type, post_format = _invert_frames(
            frames,
            in_type=in_type,
            in_format=args.from_format,
            axis=args.axis,
            verbose=args.verbose,
        )

    converted = _convert_frames(
        frames,
        in_type=post_type,
        out_type=out_type,
        in_format=post_format,
        out_format=args.to_format,
        axis=args.axis,
    )

    write_output(converted, args.output, out_type, parser)
    print("Done.")
