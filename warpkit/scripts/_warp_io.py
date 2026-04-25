"""Shared IO helpers for the warpkit warp-CLI scripts.

Both ``wk-convert-warp`` and ``wk-compute-jacobian`` accept the same
"1+ files of maps or fields" input model and the same "1 bundled file or N
per-frame files" output model. This module hosts the frame splitting and
bundling helpers that the scripts share.
"""

from __future__ import annotations

import argparse
from typing import cast

import nibabel as nib
import numpy as np


def read_input_frames(
    input_paths: list[str],
    from_type: str,
    parser: argparse.ArgumentParser,
) -> list[nib.Nifti1Image]:
    """Load input file(s) and split into a flat list of single-frame images.

    ``from_type`` is ``"map"`` (1-channel) or ``"field"`` (3-channel) — the
    user-supplied input type from ``--from``. Each input may be a single 3D
    map, a 4D map series, a 4D field, or a 5D field (singleton or
    multi-frame). The returned frames are 3D for maps and 4D
    ``(X, Y, Z, 3)`` for fields.
    """
    frames: list[nib.Nifti1Image] = []
    for p in input_paths:
        img = cast(nib.Nifti1Image, nib.load(p))
        if from_type == "map":
            if img.ndim == 3:
                frames.append(img)
            elif img.ndim == 4:
                for i in range(img.shape[-1]):
                    fd = np.asarray(img.dataobj[..., i])
                    frames.append(nib.Nifti1Image(fd, img.affine, img.header))
            else:
                parser.error(
                    f"map input must be 3D or 4D; got shape {img.shape} for {p}"
                )
        else:  # field
            if img.ndim == 4 and img.shape[-1] == 3:
                frames.append(img)
            elif img.ndim == 5 and img.shape[-1] == 3:
                # ANTs/AFNI single = (X, Y, Z, 1, 3); series = (X, Y, Z, T, 3)
                for i in range(img.shape[3]):
                    fd = np.asarray(img.dataobj[..., i, :])
                    frames.append(nib.Nifti1Image(fd, img.affine, img.header))
            else:
                parser.error(
                    "field input must be 4D (X,Y,Z,3) or 5D (X,Y,Z,T,3); "
                    f"got shape {img.shape} for {p}"
                )
    return frames


def bundle_frames_to_3d_series(frames: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    """Stack 3D scalar frames into a 4D ``(X, Y, Z, T)`` series.

    Used for both 1-channel displacement maps and scalar Jacobian fields.
    The frame headers may have inherited a vector intent code from an upstream
    field operation; clear it so downstream tools don't misread the bundled
    scalar series as a field.
    """
    data = np.stack([f.get_fdata() for f in frames], axis=-1).astype(np.float32)
    header = cast(nib.Nifti1Header, frames[0].header.copy())
    header.set_intent("none", (), "")
    return nib.Nifti1Image(data, frames[0].affine, header)


def bundle_frames_to_field_series(frames: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    """Stack 4D ``(X, Y, Z, 3)`` field frames into a 5D ``(X, Y, Z, T, 3)``
    series."""
    arrs: list[np.ndarray] = []
    for f in frames:
        a = f.get_fdata()
        if a.ndim == 5 and a.shape[3] == 1:
            a = a[..., 0, :]
        arrs.append(a)
    data = np.stack(arrs, axis=-2).astype(np.float32)
    out = nib.Nifti1Image(data, frames[0].affine, frames[0].header)
    cast(nib.Nifti1Header, out.header).set_intent("vector", (), "")
    return out


def write_output(
    frames: list[nib.Nifti1Image],
    out_paths: list[str],
    out_type: str,
    parser: argparse.ArgumentParser,
) -> None:
    """Write per-frame images either bundled into one file (when exactly one
    output path is given for >1 frames) or one file per frame.

    ``out_type`` is ``"map"`` or ``"field"`` for warp-style outputs; the
    Jacobian script passes ``"map"`` since a scalar Jacobian is shaped like
    a 1-channel map.
    """
    n = len(frames)
    n_out = len(out_paths)
    if n_out == 1 and n > 1:
        bundled = (
            bundle_frames_to_3d_series(frames)
            if out_type == "map"
            else bundle_frames_to_field_series(frames)
        )
        bundled.to_filename(out_paths[0])
    elif n_out == n:
        for path, img in zip(out_paths, frames, strict=True):
            # Per-frame map outputs may carry a vector intent inherited from an
            # upstream field operation — drop it so downstream tools don't
            # misread the file as a field.
            if out_type == "map":
                header = cast(nib.Nifti1Header, img.header.copy())
                header.set_intent("none", (), "")
                img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, header)
            img.to_filename(path)
    else:
        parser.error(
            f"got {n_out} --output path(s) for {n} frame(s); must be 1 "
            f"(bundle into a single file) or {n} (one per frame)"
        )
