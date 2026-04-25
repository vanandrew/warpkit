import argparse
from collections.abc import Callable
from typing import cast

import nibabel as nib
import numpy as np

from warpkit import __version__
from warpkit.utilities import (
    AXIS_MAP,
    WARP_ITK_FLIPS,
    convert_warp,
    displacement_map_to_field,
    resample_image,
    setup_logging,
)

from . import epilog


def _classify_single_transform(img: nib.Nifti1Image, override: str) -> str:
    """Classify a single-file transform as 'map' (1-channel) or 'field' (3-channel)."""
    if override != "auto":
        return override
    intent = img.header.get_intent() if img.header is not None else (None,)
    if intent and intent[0] == "vector":
        return "field"
    if img.ndim == 5:
        return "field"
    if img.ndim == 4 and img.shape[-1] == 3:
        return "field"
    return "map"


def _build_transform_getter(
    transforms: list[nib.Nifti1Image],
    transform_type_override: str,
    phase_encoding_axis: str | None,
    in_format: str,
    parser: argparse.ArgumentParser,
) -> tuple[int, str, Callable[[int], nib.Nifti1Image]]:
    """Validate and wrap the user-supplied transform inputs.

    Returns (frame_count, classified_type, getter). The getter is a callable
    that takes a 0-indexed frame number and returns an itk-format
    ``Nifti1Image`` ready for ``resample_image``. Single-frame transforms are
    cached on first access.
    """
    if len(transforms) > 1:
        for t in transforms:
            if t.ndim != 4 or t.shape[-1] != 3:
                parser.error(
                    "when --transform is a series, each file must be a 4D "
                    f"3-channel field (X,Y,Z,3); got shape {t.shape}"
                )
        cache: list[nib.Nifti1Image | None] = [None] * len(transforms)

        def get_series(i: int) -> nib.Nifti1Image:
            entry = cache[i]
            if entry is None:
                entry = convert_warp(transforms[i], in_type=in_format, out_type="itk")
                cache[i] = entry
            return entry

        return len(transforms), "field", get_series

    t = transforms[0]
    transform_type = _classify_single_transform(t, transform_type_override)

    if transform_type == "map":
        if not phase_encoding_axis:
            parser.error(
                "--phase-encoding-axis is required when the transform is a "
                "1-channel displacement map."
            )
        axis = cast(str, phase_encoding_axis)
        if t.ndim == 3:
            cached = displacement_map_to_field(t, axis=axis, format="itk", frame=0)
            return 1, "map", lambda _i: cached
        if t.ndim == 4:
            n = t.shape[-1]
            if n == 1:
                cached = displacement_map_to_field(t, axis=axis, format="itk", frame=0)
                return 1, "map", lambda _i: cached
            return (
                n,
                "map",
                lambda i: displacement_map_to_field(
                    t, axis=axis, format="itk", frame=i
                ),
            )
        parser.error(
            f"displacement map must be 3D or 4D; got {t.ndim}D shape {t.shape}"
        )

    # field
    if t.ndim == 4 and t.shape[-1] == 3:
        cached = convert_warp(t, in_type=in_format, out_type="itk")
        return 1, "field", lambda _i: cached
    if t.ndim == 5 and t.shape[-1] == 3:
        if t.shape[3] == 1:
            # ANTs / AFNI single-field convention (X,Y,Z,1,3)
            cached = convert_warp(t, in_type=in_format, out_type="itk")
            return 1, "field", lambda _i: cached
        n = t.shape[3]

        def get_5d_frame(i: int) -> nib.Nifti1Image:
            frame_data = np.asarray(t.dataobj[..., i, :])
            frame_img = nib.Nifti1Image(frame_data, t.affine, t.header)
            return convert_warp(frame_img, in_type=in_format, out_type="itk")

        return n, "field", get_5d_frame
    parser.error(
        "displacement field must be 4D (X,Y,Z,3) or 5D (X,Y,Z,T,3); "
        f"got shape {t.shape}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resample an image through a displacement transform. Supports "
            "single-frame and time-series resampling with either 1-channel "
            "displacement maps (warpkit/medic output, along a single phase-"
            "encoding axis) or 3-channel displacement fields "
            "(ITK/FSL/ANTs/AFNI). When the transform has N frames, frame i "
            "of the input is resampled with frame i of the transform; a "
            "single-frame transform broadcasts across all input frames."
        ),
        epilog=f"{epilog} 04/24/2026",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image to resample (3D or 4D).",
    )
    parser.add_argument(
        "--reference",
        help="Reference 3D grid for the output. Defaults to the input (frame 0 if 4D).",
    )
    parser.add_argument(
        "--transform",
        nargs="+",
        required=True,
        help=(
            "One or more displacement transforms. A single file is auto-"
            "classified as displacement maps (1-channel) or a displacement "
            "field (3-channel); pass multiple 4D fields (X,Y,Z,3) to apply "
            "a per-frame field series to a 4D input."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NIfTI path.",
    )
    parser.add_argument(
        "--transform-type",
        choices=("auto", "map", "field"),
        default="auto",
        help=(
            "Override the single-file classifier. 'map' = 1-channel "
            "displacement magnitudes along --phase-encoding-axis. 'field' = "
            "3-channel displacement vectors. Ignored when --transform has "
            "more than one file (always treated as a field series)."
        ),
    )
    parser.add_argument(
        "--phase-encoding-axis",
        choices=list(AXIS_MAP),
        metavar="AXIS",
        help=(
            "Axis the 1-channel displacement maps are along (one of: "
            f"{', '.join(AXIS_MAP)}). Required when the transform is a "
            "displacement map."
        ),
    )
    parser.add_argument(
        "--format",
        choices=list(WARP_ITK_FLIPS),
        default="itk",
        help="Format of the input transform when it is a 3-channel field (default: itk).",
    )

    args = parser.parse_args()

    setup_logging()
    print(f"wk-apply-warp: {args}")

    input_img = cast(nib.Nifti1Image, nib.load(args.input))
    transforms = [cast(nib.Nifti1Image, nib.load(p)) for p in args.transform]

    # determine the reference grid
    reference_img: nib.Nifti1Image
    if args.reference:
        reference_img = cast(nib.Nifti1Image, nib.load(args.reference))
    elif input_img.ndim == 3:
        reference_img = input_img
    else:
        reference_img = nib.Nifti1Image(
            np.asarray(input_img.dataobj[..., 0]),
            input_img.affine,
            input_img.header,
        )

    # build a per-frame transform getter
    n_transform, transform_type, get_transform = _build_transform_getter(
        transforms,
        transform_type_override=args.transform_type,
        phase_encoding_axis=args.phase_encoding_axis,
        in_format=args.format,
        parser=parser,
    )

    # input frame count
    if input_img.ndim == 3:
        n_input = 1
    elif input_img.ndim == 4:
        n_input = input_img.shape[-1]
    else:
        parser.error(
            f"input image must be 3D or 4D; got {input_img.ndim}D shape {input_img.shape}"
        )

    # compatibility checks
    if n_transform > 1 and n_input == 1:
        parser.error(
            f"got a {n_transform}-frame transform but input is 3D; input "
            "must be 4D when applying a series of transforms."
        )
    if n_transform > 1 and n_transform != n_input:
        parser.error(
            f"transform has {n_transform} frame(s) but input has {n_input}; "
            "they must match (or pass a single-frame transform to broadcast)."
        )

    print(
        f"  input frames: {n_input}; transform frames: {n_transform} "
        f"(type={transform_type})"
    )

    # resample
    if n_input == 1:
        out_img = resample_image(reference_img, input_img, get_transform(0))
    else:
        out_frames = []
        for i in range(n_input):
            frame_data = np.asarray(input_img.dataobj[..., i])
            frame_img = nib.Nifti1Image(frame_data, input_img.affine, input_img.header)
            t_idx = i if n_transform > 1 else 0
            resampled = resample_image(reference_img, frame_img, get_transform(t_idx))
            out_frames.append(resampled.get_fdata())
            if (i + 1) % 10 == 0 or (i + 1) == n_input:
                print(f"  resampled frame {i + 1}/{n_input}")
        out_data = np.stack(out_frames, axis=-1).astype(np.float32)
        out_img = nib.Nifti1Image(out_data, reference_img.affine, reference_img.header)

    print(f"Saving resampled image to {args.output}...")
    out_img.to_filename(args.output)
    print("Done.")
