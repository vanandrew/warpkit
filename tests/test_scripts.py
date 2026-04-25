"""CLI argument-validation tests.

These exercise argparse + the post-parse mutex/either-or logic in
``medic.main`` and the help text of ``extract_field_from_maps.main``. They
do not touch the heavy unwrap/inversion pipeline — every test here triggers
``parser.error`` (which raises ``SystemExit(2)``) before the data load.
"""

from __future__ import annotations

import json
import sys

import pytest
from warpkit.scripts.extract_field_from_maps import main as extract_main
from warpkit.scripts.medic import main as medic_main


@pytest.fixture
def argv(monkeypatch):
    def _set(args):
        monkeypatch.setattr(sys, "argv", args)

    return _set


# ---------------------------------------------------------------------------
# medic --help / --version
# ---------------------------------------------------------------------------


def test_medic_help(argv, capsys):
    argv(["medic", "--help"])
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Multi-Echo DIstortion Correction" in out
    # dash-form flags are documented
    assert "--out-prefix" in out
    assert "--total-readout-time" in out
    assert "--phase-encoding-direction" in out


def test_medic_version(argv, capsys):
    argv(["medic", "--version"])
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "medic" in out


# ---------------------------------------------------------------------------
# medic argument validation (either-or, mutex, count match)
# ---------------------------------------------------------------------------


def test_medic_requires_acquisition_args(argv, capsys):
    argv(
        [
            "medic",
            "--magnitude",
            "m.nii",
            "--phase",
            "p.nii",
            "--out-prefix",
            "out",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "either --metadata or all of" in err
    assert "--TEs" in err
    assert "--total-readout-time" in err
    assert "--phase-encoding-direction" in err


def test_medic_metadata_and_direct_args_conflict(argv, capsys):
    argv(
        [
            "medic",
            "--magnitude",
            "m.nii",
            "--phase",
            "p.nii",
            "--out-prefix",
            "out",
            "--metadata",
            "m.json",
            "--TEs",
            "14.0",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err
    assert "--TEs" in err


def test_medic_te_count_must_match_phase_count(argv, capsys, tmp_path):
    argv(
        [
            "medic",
            "--magnitude",
            "m1.nii",
            "m2.nii",
            "--phase",
            "p1.nii",
            "p2.nii",
            "--TEs",
            "14.0",  # only 1 TE for 2 phase files
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must match" in err


def test_medic_metadata_te_count_must_match_phase_count(argv, capsys, tmp_path):
    """Same length check fires when TEs come from --metadata sidecars."""
    sidecar = tmp_path / "m1.json"
    sidecar.write_text(
        json.dumps(
            {
                "EchoTime": 0.014,
                "TotalReadoutTime": 0.05,
                "PhaseEncodingDirection": "j",
            }
        )
    )
    argv(
        [
            "medic",
            "--magnitude",
            "m.nii",
            "--phase",
            "p1.nii",
            "p2.nii",
            "--metadata",
            str(sidecar),
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must match" in err


def test_medic_phase_encoding_direction_choices_enforced(argv, capsys):
    argv(
        [
            "medic",
            "--magnitude",
            "m.nii",
            "--phase",
            "p.nii",
            "--TEs",
            "14.0",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "diagonal",  # not a valid choice
            "--out-prefix",
            "out",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


# ---------------------------------------------------------------------------
# extract_field_from_maps --help / choices
# ---------------------------------------------------------------------------


def test_extract_field_from_maps_help(argv, capsys):
    argv(["extract_field_from_maps", "--help"])
    with pytest.raises(SystemExit) as exc:
        extract_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "extracts a displacement field" in out
    # dash-form flags
    assert "--frame-number" in out
    assert "--phase-encoding-axis" in out


def test_extract_field_from_maps_rejects_bad_axis(argv, capsys):
    argv(
        [
            "extract_field_from_maps",
            "maps.nii.gz",
            "out.nii.gz",
            "--phase-encoding-axis",
            "diagonal",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        extract_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err
