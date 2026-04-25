"""``wk-apply-warp`` — resample an image through a displacement transform.

Public surface:

* :func:`apply_warp` — typed Python entry point. Returns an
  :class:`ApplyWarpResult` with the absolute path of the written NIfTI.
* :func:`main` — argparse CLI shim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
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
from ._metadata import ensure_image, ensure_images


@dataclass(frozen=True, slots=True)
class ApplyWarpResult:
    output: Path


def _build_transform_getter(
    transforms: list[nib.Nifti1Image],
    transform_type: str,
    phase_encoding_axis: str | None,
    in_format: str,
) -> tuple[int, str, Callable[[int], nib.Nifti1Image]]:
    """Validate and wrap the user-supplied transform inputs.

    Returns (frame_count, transform_type, getter). The getter takes a
    0-indexed frame number and returns an itk-format ``Nifti1Image`` ready
    for ``resample_image``. Single-frame transforms are cached on first
    access. Validation errors raise :class:`ValueError`.
    """
    if len(transforms) > 1:
        if transform_type != "field":
            raise ValueError(
                "transform_type='map' is incompatible with a multi-file "
                "transform series (each file must be a 3-channel field). "
                "Pass a single 4D map series or use transform_type='field'."
            )
        for t in transforms:
            if t.ndim != 4 or t.shape[-1] != 3:
                raise ValueError(
                    "when transform is a series, each file must be a 4D "
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

    if transform_type == "map":
        if not phase_encoding_axis:
            raise ValueError(
                "phase_encoding_axis is required when the transform is a "
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
        raise ValueError(
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
    raise ValueError(
        "displacement field must be 4D (X,Y,Z,3) or 5D (X,Y,Z,T,3); "
        f"got shape {t.shape}"
    )


def apply_warp(
    *,
    input: PathLike[str] | str | nib.Nifti1Image,
    transform: Sequence[PathLike[str] | str | nib.Nifti1Image],
    output: PathLike[str] | str,
    transform_type: str,
    reference: PathLike[str] | str | nib.Nifti1Image | None = None,
    phase_encoding_axis: str | None = None,
    format: str = "itk",
) -> ApplyWarpResult:
    """Resample ``input`` through one or more displacement transforms and
    write the result to ``output``.

    ``transform_type`` is ``"map"`` (1-channel along ``phase_encoding_axis``)
    or ``"field"`` (3-channel). A multi-file transform must be ``"field"``;
    a single-frame transform broadcasts across all frames of a 4D input.

    ``format`` is the input format of 3-channel fields and must be one of
    ``itk`` / ``fsl`` / ``ants`` / ``afni`` (default ``itk``).
    """
    if transform_type not in ("map", "field"):
        raise ValueError(
            f"transform_type must be 'map' or 'field'; got {transform_type!r}"
        )
    if format not in WARP_ITK_FLIPS:
        raise ValueError(
            f"format must be one of {tuple(WARP_ITK_FLIPS)}; got {format!r}"
        )
    if phase_encoding_axis is not None and phase_encoding_axis not in AXIS_MAP:
        raise ValueError(
            f"phase_encoding_axis must be one of {tuple(AXIS_MAP)}; "
            f"got {phase_encoding_axis!r}"
        )

    input_img = ensure_image(input)
    transforms = ensure_images(transform)

    reference_img: nib.Nifti1Image
    if reference is not None:
        reference_img = ensure_image(reference)
    elif input_img.ndim == 3:
        reference_img = input_img
    else:
        reference_img = nib.Nifti1Image(
            np.asarray(input_img.dataobj[..., 0]),
            input_img.affine,
            input_img.header,
        )

    n_transform, post_transform_type, get_transform = _build_transform_getter(
        transforms,
        transform_type=transform_type,
        phase_encoding_axis=phase_encoding_axis,
        in_format=format,
    )

    if input_img.ndim == 3:
        n_input = 1
    elif input_img.ndim == 4:
        n_input = input_img.shape[-1]
    else:
        raise ValueError(
            f"input image must be 3D or 4D; got {input_img.ndim}D shape "
            f"{input_img.shape}"
        )

    if n_transform > 1 and n_input == 1:
        raise ValueError(
            f"got a {n_transform}-frame transform but input is 3D; input "
            "must be 4D when applying a series of transforms."
        )
    if n_transform > 1 and n_transform != n_input:
        raise ValueError(
            f"transform has {n_transform} frame(s) but input has {n_input}; "
            "they must match (or pass a single-frame transform to broadcast)."
        )

    print(
        f"  input frames: {n_input}; transform frames: {n_transform} "
        f"(type={post_transform_type})"
    )

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

    output_path = Path(str(output)).resolve()
    print(f"Saving resampled image to {output_path}...")
    out_img.to_filename(str(output_path))
    print("Done.")
    return ApplyWarpResult(output=output_path)


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
            "One or more displacement transforms. A single file may be a "
            "displacement map (1-channel along --phase-encoding-axis) or a "
            "displacement field (3-channel); the type must be declared via "
            "--transform-type. Pass multiple 4D fields (X,Y,Z,3) to apply a "
            "per-frame field series to a 4D input."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NIfTI path.",
    )
    parser.add_argument(
        "--transform-type",
        choices=("map", "field"),
        required=True,
        help=(
            "Transform type. 'map' = 1-channel displacement magnitudes along "
            "--phase-encoding-axis. 'field' = 3-channel displacement vectors. "
            "Multi-file --transform must be 'field'."
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

    # Map ValueError messages onto the dash-form CLI flags so existing
    # parser.error-style error text continues to make sense to CLI users.
    try:
        apply_warp(
            input=args.input,
            transform=args.transform,
            output=args.output,
            transform_type=args.transform_type,
            reference=args.reference,
            phase_encoding_axis=args.phase_encoding_axis,
            format=args.format,
        )
    except ValueError as e:
        msg = str(e)
        msg = msg.replace("transform_type=", "--transform-type=")
        msg = msg.replace("phase_encoding_axis", "--phase-encoding-axis")
        parser.error(msg)
