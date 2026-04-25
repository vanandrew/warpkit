"""``wk-medic`` — full Multi-Echo DIstortion Correction pipeline.

This module exposes two surfaces:

* :func:`medic` — the typed Python entry point. Takes paths or in-memory
  ``Nifti1Image`` objects, runs MEDIC, writes outputs, and returns a
  :class:`MedicResult` with the absolute paths of the three NIfTIs written
  (``<prefix>_fieldmaps_native.nii``, ``_displacementmaps.nii`` and
  ``_fieldmaps.nii``). Library/integration code (e.g. nipype interfaces)
  should call this.
* :func:`main` — the argparse CLI entry point. Parses ``sys.argv``, then
  defers to :func:`medic`. ``ValueError`` from the latter is forwarded to
  ``parser.error`` so CLI behaviour (``SystemExit(2)``, ``error: ...`` on
  stderr) is preserved.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import nibabel as nib

from warpkit import __version__
from warpkit.distortion import medic as _medic_distortion
from warpkit.utilities import setup_logging

from . import epilog
from ._metadata import ensure_images, resolve_acquisition, trim_noise_frames

PE_DIRECTIONS = ("i", "j", "k", "i-", "j-", "k-", "x", "y", "z", "x-", "y-", "z-")


@dataclass(frozen=True, slots=True)
class MedicResult:
    """Absolute paths of the three NIfTIs written by :func:`medic`."""

    fieldmap_native: Path
    displacement_map: Path
    fieldmap: Path


def medic(
    *,
    phase: Sequence[PathLike[str] | str | nib.Nifti1Image],
    magnitude: Sequence[PathLike[str] | str | nib.Nifti1Image],
    out_prefix: PathLike[str] | str,
    tes: Sequence[float] | None = None,
    total_readout_time: float | None = None,
    phase_encoding_direction: str | None = None,
    metadata: Sequence[PathLike[str] | str] | None = None,
    noise_frames: int = 0,
    n_cpus: int = 4,
    wrap_limit: bool = False,
    debug: bool = False,
) -> MedicResult:
    """Run the full MEDIC pipeline and write the three output NIfTIs.

    Either pass ``metadata`` (one BIDS sidecar per echo) or all three of
    ``tes`` (ms), ``total_readout_time`` (s) and ``phase_encoding_direction``
    (``i``/``j``/``k``/``x``/``y``/``z`` with optional trailing ``-``). The
    two are mutually exclusive.

    ``noise_frames`` trims that many frames from the end of every
    phase/magnitude file before unwrapping (matches the CLI's ``-f``).

    Returns a :class:`MedicResult` with absolute paths of the three written
    NIfTIs.
    """
    if len(phase) != len(magnitude):
        raise ValueError(
            f"got {len(phase)} phase file(s) but {len(magnitude)} magnitude "
            "file(s); they must match (one mag/phase pair per echo)."
        )

    tes_ms, trt, ped = resolve_acquisition(
        metadata=metadata,
        tes=tes,
        total_readout_time=total_readout_time,
        phase_encoding_direction=phase_encoding_direction,
        require_trt_pe=True,
    )
    # require_trt_pe=True guarantees both are populated.
    assert trt is not None and ped is not None

    if len(tes_ms) != len(phase):
        raise ValueError(
            f"got {len(tes_ms)} echo time(s) but --phase has {len(phase)} "
            "file(s); they must match."
        )

    mag_data = ensure_images(magnitude)
    phase_data = ensure_images(phase)

    if noise_frames > 0:
        print(f"Removing {noise_frames} noise frames from the end of each file...")
    mag_data = trim_noise_frames(mag_data, noise_frames)
    phase_data = trim_noise_frames(phase_data, noise_frames)

    if debug:
        fmaps_native, dmaps, fmaps = _medic_distortion(
            phase_data,
            mag_data,
            tes_ms,
            trt,
            ped,
            n_cpus=n_cpus,
            border_filt=(1000, 1000),
            svd_filt=1000,
            debug=True,
            wrap_limit=wrap_limit,
        )
    else:
        fmaps_native, dmaps, fmaps = _medic_distortion(
            phase_data,
            mag_data,
            tes_ms,
            trt,
            ped,
            n_cpus=n_cpus,
            svd_filt=10,
            border_size=5,
            wrap_limit=wrap_limit,
        )

    out_prefix_str = str(out_prefix)
    print("Saving field maps and displacement maps to file...")
    fmap_native_path = Path(f"{out_prefix_str}_fieldmaps_native.nii").resolve()
    dmap_path = Path(f"{out_prefix_str}_displacementmaps.nii").resolve()
    fmap_path = Path(f"{out_prefix_str}_fieldmaps.nii").resolve()
    fmaps_native.to_filename(str(fmap_native_path))
    dmaps.to_filename(str(dmap_path))
    fmaps.to_filename(str(fmap_path))
    print("Done.")
    return MedicResult(
        fieldmap_native=fmap_native_path,
        displacement_map=dmap_path,
        fieldmap=fmap_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Echo DIstortion Correction", epilog=f"{epilog}"
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
            "BIDS-style JSON sidecar for each echo. EchoTime (s) is read per file; "
            "TotalReadoutTime and PhaseEncodingDirection are taken from the first. "
            "Mutually exclusive with --TEs / --total-readout-time / "
            "--phase-encoding-direction."
        ),
    )
    parser.add_argument(
        "--TEs",
        dest="tes",
        nargs="+",
        type=float,
        help=(
            "Echo times in milliseconds, one per echo (must match --phase order). "
            "Required unless --metadata is given."
        ),
    )
    parser.add_argument(
        "--total-readout-time",
        type=float,
        help="Total readout time in seconds. Required unless --metadata is given.",
    )
    parser.add_argument(
        "--phase-encoding-direction",
        choices=PE_DIRECTIONS,
        metavar="DIR",
        help=(
            f"Phase encoding direction (one of: {', '.join(PE_DIRECTIONS)}). "
            "Required unless --metadata is given."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output field maps and displacement maps.",
    )
    parser.add_argument(
        "-f", "--noiseframes", type=int, default=0, help="Number of noise frames"
    )
    parser.add_argument(
        "-n", "--n-cpus", type=int, default=4, help="Number of CPUs to use."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--wrap-limit",
        action="store_true",
        help="Turns off some heuristics for phase unwrapping",
    )

    args = parser.parse_args()
    setup_logging()
    print(f"medic: {args}")

    try:
        medic(
            phase=args.phase,
            magnitude=args.magnitude,
            out_prefix=args.out_prefix,
            tes=args.tes,
            total_readout_time=args.total_readout_time,
            phase_encoding_direction=args.phase_encoding_direction,
            metadata=args.metadata,
            noise_frames=args.noiseframes,
            n_cpus=args.n_cpus,
            wrap_limit=args.wrap_limit,
            debug=args.debug,
        )
    except ValueError as e:
        parser.error(str(e))
