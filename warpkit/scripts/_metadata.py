"""Shared acquisition-metadata helpers for the warpkit script entry points.

Centralises:

* coercing user-supplied images (``Path`` / ``str`` / ``Nifti1Image``) into
  ``Nifti1Image`` objects,
* loading echo time / total readout time / phase encoding direction from
  BIDS-style JSON sidecars,
* the "either ``--metadata`` or direct args" mutex/either-or check shared by
  ``wk-medic``, ``wk-unwrap-phase``, and ``wk-compute-fieldmap``.

Validation errors raise :class:`ValueError`; the CLI shims forward those to
``parser.error`` so the user-visible behaviour (``SystemExit(2)``) is
preserved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from os import PathLike
from typing import cast

import nibabel as nib


def ensure_image(x: PathLike[str] | str | nib.Nifti1Image) -> nib.Nifti1Image:
    """Coerce a path or in-memory image into a ``Nifti1Image``."""
    if isinstance(x, nib.Nifti1Image):
        return x
    return cast(nib.Nifti1Image, nib.load(str(x)))


def ensure_images(
    xs: Sequence[PathLike[str] | str | nib.Nifti1Image],
) -> list[nib.Nifti1Image]:
    return [ensure_image(x) for x in xs]


def load_acquisition_from_metadata(
    metadata_paths: Sequence[PathLike[str] | str],
    *,
    require_trt_pe: bool = True,
) -> tuple[list[float], float | None, str | None]:
    """Read EchoTime (s → ms) — and optionally TotalReadoutTime (s) and
    PhaseEncodingDirection — from BIDS-style JSON sidecars.

    Per-echo ``EchoTime`` is read from each file; ``TotalReadoutTime`` and
    ``PhaseEncodingDirection`` are taken from the first. When
    ``require_trt_pe=False`` the latter two are skipped so callers that only
    need echo times (e.g. ``unwrap_phase``) accept sidecars that omit them.
    Missing required keys raise :class:`ValueError`.
    """
    metadatas = []
    for j in metadata_paths:
        with open(j) as f:
            metadatas.append(json.load(f))
    try:
        tes_ms = [float(m["EchoTime"]) * 1000 for m in metadatas]
    except KeyError:
        raise ValueError(
            "metadata sidecar is missing required key: 'EchoTime'."
        ) from None
    if not require_trt_pe:
        return tes_ms, None, None
    missing = [
        k
        for k in ("TotalReadoutTime", "PhaseEncodingDirection")
        if k not in metadatas[0]
    ]
    if missing:
        raise ValueError(
            "metadata sidecar is missing required key(s): "
            f"{', '.join(repr(k) for k in missing)}."
        )
    trt = float(metadatas[0]["TotalReadoutTime"])
    ped = str(metadatas[0]["PhaseEncodingDirection"])
    return tes_ms, trt, ped


def resolve_acquisition(
    *,
    metadata: Sequence[PathLike[str] | str] | None,
    tes: Sequence[float] | None,
    total_readout_time: float | None = None,
    phase_encoding_direction: str | None = None,
    require_trt_pe: bool = True,
) -> tuple[list[float], float | None, str | None]:
    """Resolve echo times / TRT / PED from either BIDS metadata or direct args.

    Mirrors the either-or / mutex logic in the CLI scripts. Set
    ``require_trt_pe=False`` for callers that only need echo times (e.g.
    ``unwrap_phase``); the error message then matches that script's wording.

    Error messages reference the dash-form CLI flags (``--metadata``,
    ``--TEs``, ...) so CLI tests continue to match; nipype/library callers
    will see the same text via ``ValueError``.
    """
    flag_map = {
        "tes": "--TEs",
        "total_readout_time": "--total-readout-time",
        "phase_encoding_direction": "--phase-encoding-direction",
    }
    if require_trt_pe:
        direct_vals = {
            "tes": tes,
            "total_readout_time": total_readout_time,
            "phase_encoding_direction": phase_encoding_direction,
        }
    else:
        direct_vals = {"tes": tes}
    direct_supplied = [k for k, v in direct_vals.items() if v is not None]

    if metadata is not None and direct_supplied:
        names = ", ".join(flag_map[k] for k in direct_supplied)
        raise ValueError(
            f"--metadata is mutually exclusive with {names}; pass one or the "
            "other, not both."
        )
    if metadata is None and len(direct_supplied) != len(direct_vals):
        if require_trt_pe:
            missing = [flag_map[k] for k in direct_vals if k not in direct_supplied]
            raise ValueError(
                "either --metadata or all of --TEs, --total-readout-time, and "
                f"--phase-encoding-direction must be provided (missing: {', '.join(missing)})."
            )
        raise ValueError("either --metadata or --TEs must be provided.")

    if metadata is not None:
        return load_acquisition_from_metadata(metadata, require_trt_pe=require_trt_pe)

    return list(tes or []), total_readout_time, phase_encoding_direction


def trim_noise_frames(images: list[nib.Nifti1Image], n: int) -> list[nib.Nifti1Image]:
    """Trim the last ``n`` frames from each 4D image. Returns the input list
    unchanged when ``n == 0``.

    When ``n > 0`` each image must be 4D with strictly more than ``n`` frames;
    otherwise ``[..., :-n]`` would either chop the Z dimension of a 3D volume
    or yield an empty 4D series that crashes downstream consumers. Both raise
    :class:`ValueError`.
    """
    if n == 0:
        return images
    if n < 0:
        raise ValueError(f"noise_frames must be non-negative; got {n}.")
    for idx, img in enumerate(images):
        if img.ndim != 4:
            raise ValueError(
                f"noise_frames={n} requires 4D images; image #{idx} has "
                f"ndim={img.ndim}."
            )
        n_frames = img.shape[-1]
        if n >= n_frames:
            raise ValueError(
                f"noise_frames={n} would leave 0 frames in image #{idx} "
                f"(has {n_frames} frame(s))."
            )
    return [
        nib.Nifti1Image(img.dataobj[..., :-n], img.affine, img.header) for img in images
    ]
