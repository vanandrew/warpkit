"""``wk-unwrap-phase`` — ROMEO multi-echo phase unwrapping (the unwrap stage of
MEDIC).

Public surface:

* :func:`unwrap_phase` — typed Python entry point. Returns an
  :class:`UnwrapPhaseResult` with the per-echo unwrapped-phase paths and the
  per-frame masks path.
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
from warpkit.unwrap import unwrap_phases
from warpkit.utilities import setup_logging

from . import epilog
from ._metadata import ensure_images, resolve_acquisition, trim_noise_frames


@dataclass(frozen=True, slots=True)
class UnwrapPhaseResult:
    unwrapped: list[Path]
    masks: Path


def unwrap_phase(
    *,
    phase: Sequence[PathLike[str] | str | nib.Nifti1Image],
    magnitude: Sequence[PathLike[str] | str | nib.Nifti1Image],
    out_prefix: PathLike[str] | str,
    tes: Sequence[float] | None = None,
    metadata: Sequence[PathLike[str] | str] | None = None,
    noise_frames: int = 0,
    n_cpus: int = 4,
    wrap_limit: bool = False,
    debug: bool = False,
) -> UnwrapPhaseResult:
    """Run ROMEO multi-echo phase unwrapping.

    Either pass ``metadata`` (one BIDS sidecar per echo) or ``tes`` directly
    (echo times in ms). The two are mutually exclusive.

    Returns absolute paths of one unwrapped-phase NIfTI per echo
    (``<prefix>_unwrapped_echo-NN.nii``) plus the per-frame masks NIfTI
    (``<prefix>_masks.nii``).
    """
    if len(magnitude) != len(phase):
        raise ValueError(
            f"got {len(magnitude)} magnitude file(s) but {len(phase)} phase "
            "file(s); they must match (one mag/phase pair per echo)."
        )
    if metadata is not None and len(metadata) != len(phase):
        raise ValueError(
            f"got {len(metadata)} metadata file(s) but {len(phase)} phase "
            "file(s); they must match (one sidecar per echo)."
        )

    tes_ms, _, _ = resolve_acquisition(metadata=metadata, tes=tes, require_trt_pe=False)

    if len(tes_ms) != len(phase):
        raise ValueError(
            f"got {len(tes_ms)} echo time(s) but --phase has {len(phase)} "
            "file(s); they must match."
        )

    mag_data = ensure_images(magnitude)
    phase_data = ensure_images(phase)

    if noise_frames < 0:
        raise ValueError(f"noise_frames must be non-negative; got {noise_frames}.")
    if noise_frames > 0:
        for label, imgs in (("phase", phase_data), ("magnitude", mag_data)):
            for idx, img in enumerate(imgs):
                n_frames = img.shape[-1] if img.ndim == 4 else 1
                if noise_frames >= n_frames:
                    raise ValueError(
                        f"noise_frames={noise_frames} would leave 0 frames "
                        f"in {label} image #{idx} (has {n_frames} frame(s))."
                    )
        print(f"Removing {noise_frames} noise frames from the end of each file...")
    mag_data = trim_noise_frames(mag_data, noise_frames)
    phase_data = trim_noise_frames(phase_data, noise_frames)

    unwrapped_imgs, masks_img = unwrap_phases(
        phase_data,
        mag_data,
        list(tes_ms),
        n_cpus=n_cpus,
        debug=debug,
        wrap_limit=wrap_limit,
    )

    out_prefix_str = str(out_prefix)
    print("Saving unwrapped phase images and masks to file...")
    unwrapped_paths: list[Path] = []
    for i_echo, img in enumerate(unwrapped_imgs, start=1):
        out_path = Path(f"{out_prefix_str}_unwrapped_echo-{i_echo:02d}.nii").resolve()
        img.to_filename(str(out_path))
        unwrapped_paths.append(out_path)
    masks_path = Path(f"{out_prefix_str}_masks.nii").resolve()
    masks_img.to_filename(str(masks_path))
    print("Done.")
    return UnwrapPhaseResult(unwrapped=unwrapped_paths, masks=masks_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ROMEO multi-echo phase unwrapping (the unwrap stage of MEDIC). "
            "Outputs one unwrapped phase NIfTI per echo plus the per-frame "
            "automask. Pair with `wk-compute-fieldmap` to obtain a "
            "native-space B0 field map."
        ),
        epilog=f"{epilog} 04/24/2026",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--magnitude", nargs="+", required=True, help="Magnitude data")
    parser.add_argument("--phase", nargs="+", required=True, help="Phase data")
    parser.add_argument(
        "--metadata",
        nargs="+",
        help=(
            "BIDS-style JSON sidecar for each echo. EchoTime (s) is read per "
            "file. Mutually exclusive with --TEs."
        ),
    )
    parser.add_argument(
        "--TEs",
        dest="tes",
        nargs="+",
        type=float,
        help=(
            "Echo times in milliseconds, one per echo (must match --phase "
            "order). Required unless --metadata is given."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output unwrapped phase and mask files.",
    )
    parser.add_argument(
        "-f", "--noiseframes", type=int, default=0, help="Number of noise frames"
    )
    parser.add_argument(
        "-n", "--n-cpus", type=int, default=4, help="Number of CPUs to use."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Skip the temporal consistency pass and dump intermediate files.",
    )
    parser.add_argument(
        "--wrap-limit",
        action="store_true",
        help="Turn off some heuristics for phase unwrapping.",
    )

    args = parser.parse_args()
    setup_logging()
    print(f"wk-unwrap-phase: {args}")

    try:
        unwrap_phase(
            phase=args.phase,
            magnitude=args.magnitude,
            out_prefix=args.out_prefix,
            tes=args.tes,
            metadata=args.metadata,
            noise_frames=args.noiseframes,
            n_cpus=args.n_cpus,
            wrap_limit=args.wrap_limit,
            debug=args.debug,
        )
    except ValueError as e:
        parser.error(str(e))
