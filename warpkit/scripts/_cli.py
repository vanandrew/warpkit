"""Shared argparse argument builders for the warpkit CLI scripts.

These cover the argument blocks that are byte-identical across multiple
scripts (``--version``, ``-n/--n-cpus``, and the EPI ``--total-readout-time`` /
``--phase-encoding-direction`` pair). Args whose help text is worded per
script (``--magnitude``, ``--metadata``, ``--debug``, ...) stay inline in each
script so their wording can diverge freely.
"""

from __future__ import annotations

import argparse

from warpkit import __version__
from warpkit.utilities import PE_DIRECTIONS


def add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--version`` action."""
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )


def add_n_cpus_arg(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``-n/--n-cpus`` option (default 4)."""
    parser.add_argument(
        "-n", "--n-cpus", type=int, default=4, help="Number of CPUs to use."
    )


def add_trt_pe_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--total-readout-time`` and ``--phase-encoding-direction``.

    Used by the pipeline scripts where both are optional (resolved from
    ``--metadata`` when omitted).
    """
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
