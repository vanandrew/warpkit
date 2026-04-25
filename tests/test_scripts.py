"""CLI argument-validation tests.

These exercise argparse + the post-parse mutex/either-or logic in
``medic.main``, ``unwrap_phase.main``, ``compute_fieldmap.main``,
``apply_warp.main`` and ``convert_warp.main``. They do not touch the heavy
unwrap/inversion/resample pipeline — every test here triggers
``parser.error`` (which raises ``SystemExit(2)``) before the data load (or
before the resample, in apply_warp's case where reaching parser.error
requires loading the inputs first).
"""

from __future__ import annotations

import json
import sys
from typing import cast

import nibabel as nib
import numpy as np
import pytest
from warpkit.scripts.apply_warp import main as apply_warp_main
from warpkit.scripts.compute_fieldmap import main as compute_fieldmap_main
from warpkit.scripts.compute_jacobian import main as compute_jacobian_main
from warpkit.scripts.convert_fieldmap import main as convert_fieldmap_main
from warpkit.scripts.convert_warp import main as convert_warp_main
from warpkit.scripts.medic import main as medic_main
from warpkit.scripts.unwrap_phase import main as unwrap_phase_main


def _load(path) -> nib.Nifti1Image:
    """Type-narrowed nib.load for tests; the bundled data and our synthetic
    NIfTIs are concretely Nifti1Image."""
    return cast(nib.Nifti1Image, nib.load(str(path)))


@pytest.fixture
def argv(monkeypatch):
    def _set(args):
        monkeypatch.setattr(sys, "argv", args)

    return _set


def _write_nifti(path, shape, affine=None, dtype=np.float32):
    """Write a zero-filled NIfTI of the requested shape; return the path."""
    if affine is None:
        affine = np.eye(4)
    nib.Nifti1Image(np.zeros(shape, dtype=dtype), affine).to_filename(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# medic --help / --version
# ---------------------------------------------------------------------------


def test_medic_help(argv, capsys):
    argv(["wk-medic", "--help"])
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
    argv(["wk-medic", "--version"])
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "wk-medic" in out


# ---------------------------------------------------------------------------
# medic argument validation (either-or, mutex, count match)
# ---------------------------------------------------------------------------


def test_medic_requires_acquisition_args(argv, capsys):
    argv(
        [
            "wk-medic",
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
            "wk-medic",
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
            "wk-medic",
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
            "wk-medic",
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
            "wk-medic",
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
# wk-convert-warp --help / argument validation
# ---------------------------------------------------------------------------


def test_convert_warp_help(argv, capsys):
    argv(["wk-convert-warp", "--help"])
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Interconvert" in out
    assert "--from" in out
    assert "--to" in out
    assert "--from-format" in out
    assert "--to-format" in out
    assert "--axis" in out
    assert "--frame" in out
    assert "--invert" in out


def test_convert_warp_rejects_bad_format(argv, capsys, tmp_path):
    field = _write_nifti(tmp_path / "field.nii", (4, 4, 4, 3))
    argv(
        [
            "wk-convert-warp",
            "--input",
            field,
            "--output",
            str(tmp_path / "out.nii"),
            "--to-format",
            "matlab",  # not a valid choice
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_convert_warp_rejects_bad_axis(argv, capsys, tmp_path):
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-convert-warp",
            "--input",
            maps,
            "--output",
            str(tmp_path / "out.nii"),
            "--to",
            "field",
            "--axis",
            "diagonal",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_convert_warp_requires_axis_for_map_to_field(argv, capsys, tmp_path):
    """Map-to-field conversion requires --axis."""
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-convert-warp",
            "--input",
            maps,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
            "--to",
            "field",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--axis is required" in err


def test_convert_warp_requires_axis_for_field_to_map(argv, capsys, tmp_path):
    """Field-to-map conversion requires --axis."""
    field = _write_nifti(tmp_path / "field.nii", (4, 4, 4, 3))
    argv(
        [
            "wk-convert-warp",
            "--input",
            field,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "field",
            "--to",
            "map",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--axis is required" in err


def test_convert_warp_frame_out_of_range(argv, capsys, tmp_path):
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-convert-warp",
            "--input",
            maps,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
            "--frame",
            "10",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "out of range" in err


def test_convert_warp_output_count_mismatch(argv, capsys, tmp_path):
    """Output paths must equal frame count or be a single bundle path."""
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))  # 5 frames
    argv(
        [
            "wk-convert-warp",
            "--input",
            maps,
            "--output",
            str(tmp_path / "o1.nii"),
            str(tmp_path / "o2.nii"),  # only 2 outputs for 5 frames
            "--from",
            "map",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must be 1" in err and "one per frame" in err


def test_convert_warp_invert_single_map_requires_axis(argv, capsys, tmp_path):
    """Inverting a single-frame map always requires --axis (the map's own axis),
    even when the output is also a map (no map<->field conversion)."""
    single_map = _write_nifti(tmp_path / "map.nii", (4, 4, 4))  # 3D = single map
    argv(
        [
            "wk-convert-warp",
            "--input",
            single_map,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
            "--invert",
            # no --axis, no --to=field
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--axis is required" in err


def test_convert_warp_invert_multi_frame_field_requires_axis(argv, capsys, tmp_path):
    """Multi-frame inversion routes through the 1D map inverter, which needs
    --axis even for a field input (to know which channel to invert along)."""
    # 4 separate 4D fields = a 4-frame field series
    fields = [_write_nifti(tmp_path / f"f{i}.nii", (4, 4, 4, 3)) for i in range(4)]
    argv(
        [
            "wk-convert-warp",
            "--input",
            *fields,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "field",
            "--invert",
            # no --axis
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--axis is required" in err
    assert "multi-frame" in err


# ---------------------------------------------------------------------------
# unwrap_phase --help / argument validation
# ---------------------------------------------------------------------------


def test_unwrap_phase_help(argv, capsys):
    argv(["wk-unwrap-phase", "--help"])
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "phase unwrapping" in out
    assert "--out-prefix" in out
    # readout/PE flags are MEDIC-only; unwrap_phase should not advertise them
    assert "--total-readout-time" not in out
    assert "--phase-encoding-direction" not in out


def test_unwrap_phase_requires_tes(argv, capsys):
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            "m.nii",
            "--phase",
            "p.nii",
            "--out-prefix",
            "out",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "either --metadata or --TEs" in err


def test_unwrap_phase_metadata_and_tes_conflict(argv, capsys):
    argv(
        [
            "wk-unwrap-phase",
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
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_unwrap_phase_te_count_must_match_phase_count(argv, capsys, tmp_path):
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            "m1.nii",
            "m2.nii",
            "--phase",
            "p1.nii",
            "p2.nii",
            "--TEs",
            "14.0",  # only 1 TE for 2 phase files
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must match" in err


def test_unwrap_phase_magnitude_count_must_match_phase_count(argv, capsys, tmp_path):
    """Mismatched --magnitude / --phase counts should raise a clean parser
    error, not a downstream zip(strict=True) ValueError."""
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            "m1.nii",  # only 1 mag for 2 phase files
            "--phase",
            "p1.nii",
            "p2.nii",
            "--TEs",
            "14.0",
            "28.0",
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--magnitude" in err and "--phase" in err


def test_unwrap_phase_metadata_count_must_match_phase_count(argv, capsys, tmp_path):
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            "m1.nii",
            "m2.nii",
            "--phase",
            "p1.nii",
            "p2.nii",
            "--metadata",
            "e1.json",  # only 1 sidecar for 2 echoes
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--metadata" in err and "--phase" in err


def test_unwrap_phase_noiseframes_negative(argv, capsys, tmp_path):
    """A negative --noiseframes must fail with a clean parser error."""
    mag = _write_nifti(tmp_path / "m.nii", (4, 4, 4, 5))
    phase = _write_nifti(tmp_path / "p.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            mag,
            "--phase",
            phase,
            "--TEs",
            "14.0",
            "--out-prefix",
            str(tmp_path / "out"),
            "-f",
            "-1",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "non-negative" in err


def test_unwrap_phase_noiseframes_consumes_all_frames(argv, capsys, tmp_path):
    """--noiseframes >= n_frames would leave 0 volumes; reject with a clean
    parser.error instead of producing an empty 4D series that crashes
    unwrap_phases at frames[0]."""
    mag = _write_nifti(tmp_path / "m.nii", (4, 4, 4, 5))
    phase = _write_nifti(tmp_path / "p.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            mag,
            "--phase",
            phase,
            "--TEs",
            "14.0",
            "--out-prefix",
            str(tmp_path / "out"),
            "-f",
            "5",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "0 frames" in err


def test_unwrap_phase_metadata_accepts_echotime_only(argv, capsys, tmp_path):
    """``wk-unwrap-phase`` only needs EchoTime; sidecars without
    TotalReadoutTime / PhaseEncodingDirection must work. A mismatched phase
    count forces a clean parser.error *after* the metadata loader resolves —
    if the loader still required TRT/PED we'd see a KeyError instead."""
    sidecar = tmp_path / "m1.json"
    sidecar.write_text(json.dumps({"EchoTime": 0.014}))
    argv(
        [
            "wk-unwrap-phase",
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
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must match" in err


def test_unwrap_phase_metadata_missing_echotime(argv, capsys, tmp_path):
    """A sidecar without EchoTime must surface as a clean parser error, not a
    KeyError from the metadata loader."""
    sidecar = tmp_path / "m1.json"
    sidecar.write_text(json.dumps({}))
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            "m.nii",
            "--phase",
            "p.nii",
            "--metadata",
            str(sidecar),
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        unwrap_phase_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "EchoTime" in err


def test_medic_noiseframes_consumes_all_frames(argv, capsys, tmp_path):
    """``-f`` >= n_frames now raises a clean parser error in medic too: the
    check moved into ``trim_noise_frames`` so every caller is protected from
    silently producing an empty 4D series."""
    mag = _write_nifti(tmp_path / "m.nii", (4, 4, 4, 5))
    phase = _write_nifti(tmp_path / "p.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-medic",
            "--magnitude",
            mag,
            "--phase",
            phase,
            "--TEs",
            "14.0",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
            "--out-prefix",
            str(tmp_path / "out"),
            "-f",
            "5",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        medic_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "0 frames" in err


# ---------------------------------------------------------------------------
# compute_fieldmap --help / argument validation
# ---------------------------------------------------------------------------


def test_compute_fieldmap_help(argv, capsys):
    argv(["wk-compute-fieldmap", "--help"])
    with pytest.raises(SystemExit) as exc:
        compute_fieldmap_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "field map" in out
    assert "--unwrapped" in out
    assert "--masks" in out
    # post-unwrap distortion correction needs the readout/PE info
    assert "--total-readout-time" in out
    assert "--phase-encoding-direction" in out


def test_compute_fieldmap_requires_acquisition_args(argv, capsys):
    argv(
        [
            "wk-compute-fieldmap",
            "--magnitude",
            "m.nii",
            "--unwrapped",
            "u.nii",
            "--masks",
            "masks.nii",
            "--out-prefix",
            "out",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        compute_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "either --metadata or all of" in err
    assert "--TEs" in err
    assert "--total-readout-time" in err
    assert "--phase-encoding-direction" in err


def test_compute_fieldmap_metadata_and_direct_args_conflict(argv, capsys):
    argv(
        [
            "wk-compute-fieldmap",
            "--magnitude",
            "m.nii",
            "--unwrapped",
            "u.nii",
            "--masks",
            "masks.nii",
            "--out-prefix",
            "out",
            "--metadata",
            "m.json",
            "--TEs",
            "14.0",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        compute_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err
    assert "--TEs" in err


def test_compute_fieldmap_input_count_must_match(argv, capsys, tmp_path):
    """Mag/unwrapped/TE counts must all line up echo-for-echo."""
    argv(
        [
            "wk-compute-fieldmap",
            "--magnitude",
            "m1.nii",
            "m2.nii",
            "--unwrapped",
            "u1.nii",  # only 1 unwrapped for 2 mags
            "--masks",
            "masks.nii",
            "--TEs",
            "14.0",
            "38.0",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        compute_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must match" in err


def test_compute_fieldmap_phase_encoding_direction_choices_enforced(argv, capsys):
    argv(
        [
            "wk-compute-fieldmap",
            "--magnitude",
            "m.nii",
            "--unwrapped",
            "u.nii",
            "--masks",
            "masks.nii",
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
        compute_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


# ---------------------------------------------------------------------------
# apply_warp --help / argument validation
#
# The validation tests that need to reach `parser.error` after the data load
# (e.g. transform/input frame-count mismatch) write tiny synthetic NIfTIs to
# tmp_path so we don't depend on the heavy MEDIC fixtures.
# ---------------------------------------------------------------------------


def test_apply_warp_help(argv, capsys):
    argv(["wk-apply-warp", "--help"])
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Resample" in out
    assert "--input" in out
    assert "--transform" in out
    assert "--phase-encoding-axis" in out


def test_apply_warp_requires_phase_encoding_axis_for_map(argv, capsys, tmp_path):
    """Map-type transforms require --phase-encoding-axis."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))
    # 4D 1-channel map series
    tx = _write_nifti(tmp_path / "tx.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            tx,
            "--transform-type",
            "map",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--phase-encoding-axis" in err


def test_apply_warp_rejects_bad_axis(argv, capsys, tmp_path):
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))
    tx = _write_nifti(tmp_path / "tx.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            tx,
            "--phase-encoding-axis",
            "diagonal",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_apply_warp_rejects_bad_format(argv, capsys, tmp_path):
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))
    tx = _write_nifti(tmp_path / "tx.nii", (4, 4, 4, 3))  # 4D field
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            tx,
            "--format",
            "matlab",  # not a valid choice
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_apply_warp_3d_input_with_series_transform_errors(argv, capsys, tmp_path):
    """3D input + N-frame map transform is ambiguous — should be rejected."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))  # 3D input
    tx = _write_nifti(tmp_path / "tx.nii", (4, 4, 4, 5))  # 5-frame map series
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            tx,
            "--transform-type",
            "map",
            "--phase-encoding-axis",
            "j",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "input is 3D" in err


def test_apply_warp_frame_count_mismatch_errors(argv, capsys, tmp_path):
    """4D input frames must match transform frames when transform is a series."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4, 7))  # 7 input frames
    tx = _write_nifti(tmp_path / "tx.nii", (4, 4, 4, 5))  # 5 transform frames
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            tx,
            "--transform-type",
            "map",
            "--phase-encoding-axis",
            "j",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "they must match" in err


def test_apply_warp_series_files_must_be_3channel(argv, capsys, tmp_path):
    """Multiple --transform files are treated as a field series; each must be 4D last==3."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4, 2))
    # 4D last dim != 3 → not a valid field
    bad1 = _write_nifti(tmp_path / "bad1.nii", (4, 4, 4, 5))
    bad2 = _write_nifti(tmp_path / "bad2.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            bad1,
            bad2,
            "--transform-type",
            "field",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "4D" in err and "3-channel" in err


# ---------------------------------------------------------------------------
# CLI happy-path tests
#
# Unlike the validation tests above, these run each CLI to completion and
# assert on the output files. They use the bundled MEDIC fixture for the
# unwrap/medic/fieldmap pipeline (test_data_paths from conftest.py) and tiny
# synthetic NIfTIs for apply-warp / convert-warp (no need for real data
# there). These are the regression backstop for the script bodies that
# argparse-only tests can't reach.
# ---------------------------------------------------------------------------


def test_medic_happy_path(argv, test_data_paths, tmp_path):
    """Run wk-medic end-to-end on the bundled fixture; assert all three
    output NIfTIs exist with the expected shape and finite values."""
    out_prefix = tmp_path / "run"
    argv(
        [
            "wk-medic",
            "--magnitude",
            *test_data_paths["mag"],
            "--phase",
            *test_data_paths["phase"],
            "--metadata",
            *test_data_paths["metadata"],
            "--out-prefix",
            str(out_prefix),
            "-n",
            "1",
        ]
    )
    medic_main()
    for suffix in (
        "_fieldmaps_native.nii",
        "_displacementmaps.nii",
        "_fieldmaps.nii",
    ):
        out = _load(f"{out_prefix}{suffix}")
        data = out.get_fdata()
        assert out.ndim == 4
        assert out.shape[:3] == (64, 64, 40)
        assert out.shape[3] == 15
        assert np.isfinite(data).all()


def test_unwrap_phase_then_compute_fieldmap_happy_path(argv, test_data_paths, tmp_path):
    """Run wk-unwrap-phase, then wk-compute-fieldmap on its outputs. Together
    these cover the post-medic-split CLI flow."""
    unwrap_prefix = tmp_path / "unwrap"
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            *test_data_paths["mag"],
            "--phase",
            *test_data_paths["phase"],
            "--metadata",
            *test_data_paths["metadata"],
            "--out-prefix",
            str(unwrap_prefix),
            "-n",
            "1",
        ]
    )
    unwrap_phase_main()

    # one unwrapped phase per echo + one masks file
    unwrapped = sorted(tmp_path.glob("unwrap_unwrapped_echo-*.nii"))
    assert len(unwrapped) == 3
    masks_path = tmp_path / "unwrap_masks.nii"
    assert masks_path.exists()
    for u in unwrapped:
        img = _load(str(u))
        assert img.shape == (64, 64, 40, 15)
        assert np.isfinite(img.get_fdata()).all()

    fmap_prefix = tmp_path / "fmap"
    argv(
        [
            "wk-compute-fieldmap",
            "--magnitude",
            *test_data_paths["mag"],
            "--unwrapped",
            *[str(u) for u in unwrapped],
            "--masks",
            str(masks_path),
            "--metadata",
            *test_data_paths["metadata"],
            "--out-prefix",
            str(fmap_prefix),
            "-n",
            "1",
        ]
    )
    compute_fieldmap_main()
    for suffix in (
        "_fieldmaps_native.nii",
        "_displacementmaps.nii",
        "_fieldmaps.nii",
    ):
        out = _load(f"{fmap_prefix}{suffix}")
        assert out.shape == (64, 64, 40, 15)
        assert np.isfinite(out.get_fdata()).all()


def test_apply_warp_happy_path_zero_displacement(argv, tmp_path):
    """Identity resample: applying a zero displacement map to a non-trivial
    image gives back the same image (within ITK resampling tolerance)."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    img_data = rng.random((8, 8, 8), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    # a 4D 1-channel zero displacement map
    zero_map = np.zeros((8, 8, 8, 1), dtype=np.float32)
    map_path = tmp_path / "map.nii"
    nib.Nifti1Image(zero_map, affine).to_filename(str(map_path))

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            str(map_path),
            "--transform-type",
            "map",
            "--phase-encoding-axis",
            "j",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    out_data = out.get_fdata()
    # Output is broadcast over the single transform frame, so it carries a
    # length-1 time axis; squeeze for the value comparison.
    assert out_data.squeeze().shape == img_data.shape
    np.testing.assert_allclose(out_data.squeeze(), img_data, atol=1e-3)


def test_apply_warp_happy_path_4d_image_with_zero_field(argv, tmp_path):
    """4D BOLD-like input + single 3-channel zero field broadcasts across
    timepoints; identity resample should preserve every frame. Exercises the
    single-field branch of the transform getter and the per-frame resample
    loop."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(2)
    img_data = rng.random((8, 8, 8, 4), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    # single 4D 3-channel zero field
    zero_field = np.zeros((8, 8, 8, 3), dtype=np.float32)
    field_img = nib.Nifti1Image(zero_field, affine)
    field_img.header.set_intent("vector", (), "")
    field_path = tmp_path / "field.nii"
    field_img.to_filename(str(field_path))

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            str(field_path),
            "--transform-type",
            "field",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    assert out.shape == img_data.shape
    np.testing.assert_allclose(out.get_fdata(), img_data, atol=1e-3)


def test_apply_warp_happy_path_field_series(argv, tmp_path):
    """Multi-file --transform = a per-frame field series. Exercises the
    series branch of the transform getter."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(3)
    img_data = rng.random((8, 8, 8, 3), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    field_paths = []
    for i in range(3):
        path = tmp_path / f"field_{i}.nii"
        nib.Nifti1Image(np.zeros((8, 8, 8, 3), dtype=np.float32), affine).to_filename(
            str(path)
        )
        field_paths.append(str(path))

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            *field_paths,
            "--transform-type",
            "field",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    assert out.shape == img_data.shape
    np.testing.assert_allclose(out.get_fdata(), img_data, atol=1e-3)


def test_convert_warp_happy_path_map_field_roundtrip(argv, tmp_path):
    """Map -> field -> map round-trip preserves values within the
    convert_warp orient-roundtrip jitter."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    map_data = rng.random((6, 6, 6, 4), dtype=np.float32)
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    # maps -> bundled 5D field series (single output path)
    field_path = tmp_path / "fields.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(map_path),
            "--output",
            str(field_path),
            "--from",
            "map",
            "--to",
            "field",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()
    fields = _load(str(field_path))
    assert fields.shape == (6, 6, 6, 4, 3)
    assert fields.header.get_intent()[0] == "vector"

    # fields -> maps (single bundled output path)
    back_path = tmp_path / "maps_back.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(field_path),
            "--output",
            str(back_path),
            "--from",
            "field",
            "--to",
            "map",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()
    back = _load(str(back_path))
    assert back.shape == map_data.shape
    np.testing.assert_allclose(back.get_fdata(), map_data, atol=1e-3)


def test_convert_warp_happy_path_format_conversion(argv, tmp_path):
    """itk -> ants format conversion: ants ch0/ch1 are sign-flipped vs itk,
    output is 5D with a singleton 4th axis."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(1)
    field_data = rng.random((6, 6, 6, 3), dtype=np.float32)
    in_path = tmp_path / "field_itk.nii"
    nib.Nifti1Image(field_data, affine).to_filename(str(in_path))

    out_path = tmp_path / "field_ants.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--from",
            "field",
            "--to-format",
            "ants",
        ]
    )
    convert_warp_main()
    ants = _load(str(out_path))
    # ants single-field convention is 5D with shape (X, Y, Z, 1, 3)
    assert ants.shape == (6, 6, 6, 1, 3)
    ants_squeezed = ants.get_fdata().squeeze(axis=3)
    # ants flip array is (-1, -1, 1) — channels 0 and 1 negated, 2 unchanged
    np.testing.assert_allclose(ants_squeezed[..., 0], -field_data[..., 0], atol=1e-5)
    np.testing.assert_allclose(ants_squeezed[..., 1], -field_data[..., 1], atol=1e-5)
    np.testing.assert_allclose(ants_squeezed[..., 2], field_data[..., 2], atol=1e-5)


def test_convert_warp_happy_path_invert_zero_map(argv, tmp_path):
    """Inverting a zero displacement map gives back zero (the identity warp
    is its own inverse). Exercises the multi-frame map invert route."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    map_data = np.zeros((6, 6, 6, 3), dtype=np.float32)
    in_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(in_path))

    out_path = tmp_path / "maps_inv.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--from",
            "map",
            "--invert",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()
    inv = _load(str(out_path))
    assert inv.shape == map_data.shape
    np.testing.assert_allclose(inv.get_fdata(), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# wk-compute-jacobian — argument validation + happy paths
# ---------------------------------------------------------------------------


def test_compute_jacobian_help(argv, capsys):
    argv(["wk-compute-jacobian", "--help"])
    with pytest.raises(SystemExit) as exc:
        compute_jacobian_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Jacobian" in out
    assert "--input" in out
    assert "--axis" in out
    assert "--frame" in out


def test_compute_jacobian_requires_axis_for_map(argv, capsys, tmp_path):
    """Map inputs need --axis to promote to a 3-channel field."""
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            maps,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        compute_jacobian_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--axis is required" in err


def test_compute_jacobian_rejects_bad_format(argv, capsys, tmp_path):
    field = _write_nifti(tmp_path / "field.nii", (4, 4, 4, 3))
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            field,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "field",
            "--from-format",
            "matlab",  # not a valid choice
        ]
    )
    with pytest.raises(SystemExit) as exc:
        compute_jacobian_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_compute_jacobian_frame_out_of_range(argv, capsys, tmp_path):
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            maps,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
            "--axis",
            "j",
            "--frame",
            "10",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        compute_jacobian_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "out of range" in err


def test_compute_jacobian_single_field_zero(argv, tmp_path):
    """Jacobian of a zero displacement field is identically 1."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field_data = np.zeros((6, 6, 6, 3), dtype=np.float32)
    field_path = tmp_path / "field.nii"
    nib.Nifti1Image(field_data, affine).to_filename(str(field_path))

    out_path = tmp_path / "jdet.nii"
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(field_path),
            "--output",
            str(out_path),
            "--from",
            "field",
        ]
    )
    compute_jacobian_main()
    j = _load(str(out_path))
    assert j.shape == (6, 6, 6)
    np.testing.assert_allclose(j.get_fdata(), 1.0, atol=1e-5)


def test_compute_jacobian_map_series_zero(argv, tmp_path):
    """Multi-frame zero map series → Jacobian of 1 per frame, bundled into a
    4D output. Exercises the map-promotion path and bundle write."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    map_data = np.zeros((6, 6, 6, 4), dtype=np.float32)
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    out_path = tmp_path / "jdet.nii"
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(map_path),
            "--output",
            str(out_path),
            "--from",
            "map",
            "--axis",
            "j",
        ]
    )
    compute_jacobian_main()
    j = _load(str(out_path))
    assert j.shape == (6, 6, 6, 4)
    np.testing.assert_allclose(j.get_fdata(), 1.0, atol=1e-5)


def test_compute_jacobian_per_frame_outputs(argv, tmp_path):
    """N output paths → one Jacobian volume per frame."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    map_data = np.zeros((6, 6, 6, 4), dtype=np.float32)
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    out_paths = [str(tmp_path / f"j{i}.nii") for i in range(4)]
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(map_path),
            "--output",
            *out_paths,
            "--from",
            "map",
            "--axis",
            "j",
        ]
    )
    compute_jacobian_main()
    for p in out_paths:
        j = _load(p)
        assert j.shape == (6, 6, 6)
        np.testing.assert_allclose(j.get_fdata(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# wk-convert-fieldmap — argument validation + happy paths
# ---------------------------------------------------------------------------


def test_convert_fieldmap_help(argv, capsys):
    argv(["wk-convert-fieldmap", "--help"])
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "mm displacement" in out and "Hz" in out
    assert "--total-readout-time" in out
    assert "--phase-encoding-direction" in out
    assert "--to" in out
    assert "--flip-sign" in out


def test_convert_fieldmap_requires_to(argv, capsys, tmp_path):
    """--to is required (no sensible default)."""
    fmap = _write_nifti(tmp_path / "fmap.nii", (4, 4, 4))
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            fmap,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "fieldmap",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--to" in err


def test_convert_fieldmap_rejects_bad_pe_direction(argv, capsys, tmp_path):
    fmap = _write_nifti(tmp_path / "fmap.nii", (4, 4, 4))
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            fmap,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "fieldmap",
            "--to",
            "map",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "diagonal",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_convert_fieldmap_requires_trt_and_pe(argv, capsys, tmp_path):
    fmap = _write_nifti(tmp_path / "fmap.nii", (4, 4, 4))
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            fmap,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "fieldmap",
            "--to",
            "map",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--total-readout-time" in err
    assert "--phase-encoding-direction" in err


def test_convert_fieldmap_rejects_mm_to_mm(argv, capsys, tmp_path):
    """--from=map --to=field is a representation conversion (no unit
    crossing). The script should redirect users to wk-convert-warp."""
    maps = _write_nifti(tmp_path / "maps.nii", (4, 4, 4, 5))
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            maps,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
            "--to",
            "field",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "wk-convert-warp" in err


def test_convert_fieldmap_rejects_same_from_to(argv, capsys, tmp_path):
    fmap = _write_nifti(tmp_path / "fmap.nii", (4, 4, 4))
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            fmap,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "fieldmap",
            "--to",
            "fieldmap",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "are the same" in err


def test_convert_fieldmap_map_to_fieldmap_roundtrip(argv, tmp_path):
    """mm displacement map -> Hz fieldmap -> mm displacement map preserves
    the original (within float roundtrip jitter)."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    map_data = rng.random((6, 6, 6, 4), dtype=np.float32) - 0.5
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    fmap_path = tmp_path / "fmap.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(map_path),
            "--output",
            str(fmap_path),
            "--from",
            "map",
            "--to",
            "fieldmap",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    convert_fieldmap_main()
    fmap = _load(str(fmap_path))
    assert fmap.shape == map_data.shape
    assert np.isfinite(fmap.get_fdata()).all()

    back_path = tmp_path / "maps_back.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(fmap_path),
            "--output",
            str(back_path),
            "--from",
            "fieldmap",
            "--to",
            "map",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    convert_fieldmap_main()
    back = _load(str(back_path))
    assert back.shape == map_data.shape
    np.testing.assert_allclose(back.get_fdata(), map_data, atol=1e-5)


def test_convert_fieldmap_field_to_fieldmap(argv, tmp_path):
    """3-channel mm displacement field -> Hz fieldmap. Exercises the
    field->map axis-extraction branch upstream of the unit conversion."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(1)
    field_data = np.zeros((6, 6, 6, 3), dtype=np.float32)
    field_data[..., 1] = rng.random((6, 6, 6), dtype=np.float32) - 0.5
    field_path = tmp_path / "field.nii"
    nib.Nifti1Image(field_data, affine).to_filename(str(field_path))

    out_path = tmp_path / "fmap.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(field_path),
            "--output",
            str(out_path),
            "--from",
            "field",
            "--to",
            "fieldmap",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    convert_fieldmap_main()
    fmap = _load(str(out_path))
    assert fmap.shape == (6, 6, 6)
    assert np.isfinite(fmap.get_fdata()).all()
    # zero off-axis channels in the field -> the fieldmap should be a
    # nontrivial scalar volume (not all zeros).
    assert float(np.abs(fmap.get_fdata()).max()) > 0.0


def test_convert_fieldmap_fieldmap_to_field(argv, tmp_path):
    """Hz fieldmap -> mm 3-channel field."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(2)
    fmap_data = rng.random((6, 6, 6), dtype=np.float32) - 0.5
    fmap_path = tmp_path / "fmap.nii"
    nib.Nifti1Image(fmap_data, affine).to_filename(str(fmap_path))

    field_path = tmp_path / "field.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(fmap_path),
            "--output",
            str(field_path),
            "--from",
            "fieldmap",
            "--to",
            "field",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    convert_fieldmap_main()
    field = _load(str(field_path))
    assert field.shape == (6, 6, 6, 3)
    # j-axis has the displacement; off-axis channels are zero.
    fdata = field.get_fdata()
    assert float(np.abs(fdata[..., 1]).max()) > 0.0
    np.testing.assert_allclose(fdata[..., 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(fdata[..., 2], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# _warp_io intent-leak regression tests
# ---------------------------------------------------------------------------


def _vector_intent_frame(shape=(4, 4, 4)) -> nib.Nifti1Image:
    """Construct a 3D scalar frame whose header carries a stale vector intent
    (the typical leak from an upstream field operation)."""
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    cast(nib.Nifti1Header, img.header).set_intent("vector", (), "")
    return img


def test_bundle_frames_to_3d_series_clears_vector_intent():
    from warpkit.scripts._warp_io import bundle_frames_to_3d_series

    frames = [_vector_intent_frame() for _ in range(3)]
    bundled = bundle_frames_to_3d_series(frames)
    assert bundled.shape == (4, 4, 4, 3)
    assert bundled.header.get_intent()[0] != "vector"


def test_write_output_per_frame_map_clears_vector_intent(tmp_path):
    """Per-frame map outputs must round-trip without a stale vector intent."""
    from warpkit.scripts._warp_io import write_output

    frames = [_vector_intent_frame() for _ in range(2)]
    out_paths = [str(tmp_path / "f1.nii"), str(tmp_path / "f2.nii")]

    write_output(frames, out_paths, "map")

    for p in out_paths:
        loaded = _load(p)
        assert loaded.header.get_intent()[0] != "vector"


# ---------------------------------------------------------------------------
# CLI frame selection: --frame extracts the right frame, per-frame outputs
# preserve frame ordering.
# ---------------------------------------------------------------------------


def test_convert_warp_frame_extracts_right_frame(argv, tmp_path):
    """--frame N picks the Nth frame of a 4D map series (not just any frame)."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    map_data = np.stack(
        [np.full((4, 4, 4), float(i), dtype=np.float32) for i in range(5)],
        axis=-1,
    )
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    out_path = tmp_path / "frame.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(map_path),
            "--output",
            str(out_path),
            "--from",
            "map",
            "--frame",
            "3",
        ]
    )
    convert_warp_main()
    out = _load(str(out_path))
    np.testing.assert_allclose(out.get_fdata(), 3.0, atol=1e-6)


def test_convert_warp_per_frame_outputs_preserve_ordering(argv, tmp_path):
    """N --output paths for a 4D map series produce one file per frame with
    frame i landing in the i-th output path (not shuffled)."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    map_data = np.stack(
        [np.full((4, 4, 4), float(i), dtype=np.float32) for i in range(4)],
        axis=-1,
    )
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    out_paths = [tmp_path / f"f{i}.nii" for i in range(4)]
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(map_path),
            "--output",
            *[str(p) for p in out_paths],
            "--from",
            "map",
        ]
    )
    convert_warp_main()
    for i, p in enumerate(out_paths):
        np.testing.assert_allclose(_load(str(p)).get_fdata(), float(i), atol=1e-6)


def test_convert_fieldmap_frame_extracts_right_frame(argv, tmp_path):
    """wk-convert-fieldmap --frame N selects the Nth frame before the Hz->mm
    conversion."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    fmap_data = np.stack(
        [np.full((4, 4, 4), 10.0 * (i + 1), dtype=np.float32) for i in range(3)],
        axis=-1,
    )
    fmap_path = tmp_path / "fmap.nii"
    nib.Nifti1Image(fmap_data, affine).to_filename(str(fmap_path))

    out_path = tmp_path / "dmap.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(fmap_path),
            "--output",
            str(out_path),
            "--from",
            "fieldmap",
            "--to",
            "map",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "k",  # axis 2 → no LPS-x/y flip; positive sign for "k"
            "--frame",
            "1",
        ]
    )
    convert_fieldmap_main()
    # Frame 1 has fmap=20 Hz; voxel=2mm, trt=0.05s → expected disp = 20*0.05*2 = +2.0 mm
    np.testing.assert_allclose(_load(str(out_path)).get_fdata(), 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# CLI flip-sign behavior: --flip-sign exactly negates the no-flip output
# ---------------------------------------------------------------------------


def test_convert_fieldmap_flip_sign_exact_negation(argv, tmp_path):
    """Running map -> fieldmap twice (with and without --flip-sign) on the
    same input must produce results that are exact negations."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    map_data = (rng.standard_normal((6, 6, 6)) * 1.5).astype(np.float32)
    map_path = tmp_path / "map.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    common = [
        "--input",
        str(map_path),
        "--from",
        "map",
        "--to",
        "fieldmap",
        "--total-readout-time",
        "0.05",
        "--phase-encoding-direction",
        "j",
    ]

    out_plain = tmp_path / "fmap_plain.nii"
    argv(["wk-convert-fieldmap", *common, "--output", str(out_plain)])
    convert_fieldmap_main()
    out_flip = tmp_path / "fmap_flip.nii"
    argv(["wk-convert-fieldmap", *common, "--output", str(out_flip), "--flip-sign"])
    convert_fieldmap_main()

    np.testing.assert_allclose(
        _load(str(out_flip)).get_fdata(),
        -_load(str(out_plain)).get_fdata(),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# CLI 5D ANTs/AFNI single-warp file roundtrip
# ---------------------------------------------------------------------------


def test_convert_warp_5d_ants_file_roundtrip(argv, tmp_path):
    """A 5D ANTs single-warp file (X,Y,Z,1,3) must load through wk-convert-warp,
    convert to itk, and convert back to ants without losing data."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    ants_5d = rng.standard_normal((5, 5, 5, 1, 3)).astype(np.float32)
    in_path = tmp_path / "ants.nii"
    img = nib.Nifti1Image(ants_5d, affine)
    cast(nib.Nifti1Header, img.header).set_intent("vector", (), "")
    img.to_filename(str(in_path))

    itk_path = tmp_path / "itk.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(in_path),
            "--output",
            str(itk_path),
            "--from",
            "field",
            "--from-format",
            "ants",
            "--to-format",
            "itk",
        ]
    )
    convert_warp_main()

    back_path = tmp_path / "ants_back.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(itk_path),
            "--output",
            str(back_path),
            "--from",
            "field",
            "--from-format",
            "itk",
            "--to-format",
            "ants",
        ]
    )
    convert_warp_main()
    back = _load(str(back_path)).get_fdata()
    # Result should match the original 5D ANTs file (within orient-roundtrip noise).
    assert back.shape == ants_5d.shape
    np.testing.assert_allclose(back, ants_5d, atol=1e-5)


# ---------------------------------------------------------------------------
# CLI medic-chain reconstruction:
#   fieldmap_native (Hz, distorted) -> mm -> invert -> Hz (with --flip-sign)
# must reproduce the medic non-native fieldmap output bit-for-bit.
# ---------------------------------------------------------------------------


def test_cli_chain_reproduces_medic_non_native_fieldmap(argv, tmp_path):
    """Hz->mm->invert->Hz(--flip-sign) chain via the CLIs matches what
    warpkit.distortion.medic does internally (distortion.py:122-149).
    Skips the correlation-based safety negation, which doesn't fire on
    smooth synthetic input."""
    from scipy.ndimage import gaussian_filter
    from warpkit.utilities import (
        displacement_maps_to_field_maps,
        field_maps_to_displacement_maps,
        invert_displacement_maps,
    )

    pe = "j"
    trt = 0.05
    affine = np.diag([2.5, 2.5, 2.5, 1.0])
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.standard_normal((10, 10, 10)) * 20.0, sigma=2.0)
    fmap_data = np.stack([base, 0.95 * base], axis=-1).astype(np.float32)
    fmap_path = tmp_path / "fmap_native.nii"
    nib.Nifti1Image(fmap_data, affine).to_filename(str(fmap_path))

    # 1) Reference: medic's internal chain, frame-by-frame.
    medic_frames = []
    for i in range(fmap_data.shape[-1]):
        frame_4d = nib.Nifti1Image(fmap_data[..., i : i + 1], affine)
        inv_disp = field_maps_to_displacement_maps(frame_4d, trt, pe)
        disp = invert_displacement_maps(inv_disp, pe)
        recon = displacement_maps_to_field_maps(disp, trt, pe, flip_sign=True)
        medic_frames.append(recon.get_fdata())
    medic_out = np.stack(medic_frames, axis=-1).squeeze()

    # 2) CLI chain: Hz -> mm
    dmap_distorted = tmp_path / "dmap_distorted.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(fmap_path),
            "--output",
            str(dmap_distorted),
            "--from",
            "fieldmap",
            "--to",
            "map",
            "--total-readout-time",
            str(trt),
            "--phase-encoding-direction",
            pe,
        ]
    )
    convert_fieldmap_main()

    # 3) invert
    dmap_undistorted = tmp_path / "dmap_undistorted.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(dmap_distorted),
            "--output",
            str(dmap_undistorted),
            "--from",
            "map",
            "--invert",
            "--axis",
            pe,
        ]
    )
    convert_warp_main()

    # 4) mm -> Hz with --flip-sign
    fmap_recon = tmp_path / "fmap_recon.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(dmap_undistorted),
            "--output",
            str(fmap_recon),
            "--from",
            "map",
            "--to",
            "fieldmap",
            "--flip-sign",
            "--total-readout-time",
            str(trt),
            "--phase-encoding-direction",
            pe,
        ]
    )
    convert_fieldmap_main()

    cli_out = _load(str(fmap_recon)).get_fdata()
    assert cli_out.shape == medic_out.shape
    np.testing.assert_allclose(cli_out, medic_out, atol=1e-5)


# ---------------------------------------------------------------------------
# CLI inverse self-consistency: invert(invert(disp)) ≈ disp on small smooth disp.
# ---------------------------------------------------------------------------


def test_convert_warp_double_invert_recovers_input(argv, tmp_path):
    """Double inversion of a small smooth displacement map approximately
    recovers the original (an invertibility / discretization sanity check)."""
    from scipy.ndimage import gaussian_filter

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.standard_normal((12, 12, 12)) * 0.3, sigma=2.5)
    map_data = base.astype(np.float32)
    in_path = tmp_path / "map.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(in_path))

    inv_path = tmp_path / "map_inv.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(in_path),
            "--output",
            str(inv_path),
            "--from",
            "map",
            "--invert",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()

    inv2_path = tmp_path / "map_inv2.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(inv_path),
            "--output",
            str(inv2_path),
            "--from",
            "map",
            "--invert",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()
    recovered = _load(str(inv2_path)).get_fdata()
    # Discretization error grows with displacement magnitude; trim a 2-voxel
    # border and use a loose tolerance.
    interior = (slice(2, -2),) * 3
    np.testing.assert_allclose(recovered[interior], map_data[interior], atol=5e-2)


# ---------------------------------------------------------------------------
# wk-apply-warp coverage gaps: 3D map, 5D ANTs single field, 5D field series,
# --reference, multi-file + --transform-type=map error, non-itk --format.
# ---------------------------------------------------------------------------


def test_apply_warp_multi_file_with_transform_type_map_errors(argv, capsys, tmp_path):
    """A multi-file --transform with --transform-type=map is incompatible
    (only fields can be passed as a multi-file series)."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4, 2))
    f1 = _write_nifti(tmp_path / "f1.nii", (4, 4, 4, 3))
    f2 = _write_nifti(tmp_path / "f2.nii", (4, 4, 4, 3))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            f1,
            f2,
            "--transform-type",
            "map",
            "--phase-encoding-axis",
            "j",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "incompatible" in err and "field" in err


def test_apply_warp_invalid_map_shape_errors(argv, capsys, tmp_path):
    """A 5D --transform with --transform-type=map is rejected (maps are 3D
    or 4D scalar)."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))
    bad_map = _write_nifti(tmp_path / "bad.nii", (4, 4, 4, 1, 3))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            bad_map,
            "--transform-type",
            "map",
            "--phase-encoding-axis",
            "j",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "displacement map must be 3D or 4D" in err


def test_apply_warp_invalid_input_shape_errors(argv, capsys, tmp_path):
    """A 5D --input is rejected (input must be 3D or 4D)."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4, 2, 2))  # 5D
    tx = _write_nifti(tmp_path / "tx.nii", (4, 4, 4, 3))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            tx,
            "--transform-type",
            "field",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "input image must be 3D or 4D" in err


def test_apply_warp_3d_map_transform(argv, tmp_path):
    """A 3D single-frame displacement map (not 4D) routes through the cached
    single-frame branch."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    img_data = rng.random((6, 6, 6), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    map_path = tmp_path / "map.nii"
    nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), affine).to_filename(
        str(map_path)
    )

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            str(map_path),
            "--transform-type",
            "map",
            "--phase-encoding-axis",
            "j",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    np.testing.assert_allclose(out.get_fdata(), img_data, atol=1e-3)


def test_apply_warp_5d_ants_single_field(argv, tmp_path):
    """5D (X,Y,Z,1,3) zero ANTs single-warp file. Identity resample preserves
    the input."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    img_data = rng.random((6, 6, 6), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    field_5d = np.zeros((6, 6, 6, 1, 3), dtype=np.float32)
    field_path = tmp_path / "ants.nii"
    field_img = nib.Nifti1Image(field_5d, affine)
    cast(nib.Nifti1Header, field_img.header).set_intent("vector", (), "")
    field_img.to_filename(str(field_path))

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            str(field_path),
            "--transform-type",
            "field",
            "--format",
            "ants",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    np.testing.assert_allclose(out.get_fdata().squeeze(), img_data, atol=1e-3)


def test_apply_warp_5d_field_series_per_frame(argv, tmp_path):
    """A 5D (X,Y,Z,T,3) field series in a single file applied to a 4D input;
    each frame uses its own field. Zero fields → identity per frame."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    img_data = rng.random((5, 5, 5, 3), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    field_5d = np.zeros((5, 5, 5, 3, 3), dtype=np.float32)
    field_img = nib.Nifti1Image(field_5d, affine)
    cast(nib.Nifti1Header, field_img.header).set_intent("vector", (), "")
    field_path = tmp_path / "field5d.nii"
    field_img.to_filename(str(field_path))

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            str(field_path),
            "--transform-type",
            "field",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    assert out.shape == img_data.shape
    np.testing.assert_allclose(out.get_fdata(), img_data, atol=1e-3)


def test_apply_warp_with_explicit_reference(argv, tmp_path):
    """--reference uses an explicit grid (not the input). The output adopts
    the reference's shape and affine."""
    in_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    ref_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    ref_affine[:3, 3] = [1.0, 2.0, 3.0]  # different origin

    rng = np.random.default_rng(0)
    img = rng.random((6, 6, 6), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img, in_affine).to_filename(str(in_path))

    ref = np.zeros((4, 4, 4), dtype=np.float32)
    ref_path = tmp_path / "ref.nii"
    nib.Nifti1Image(ref, ref_affine).to_filename(str(ref_path))

    field = np.zeros((6, 6, 6, 3), dtype=np.float32)
    field_img = nib.Nifti1Image(field, in_affine)
    cast(nib.Nifti1Header, field_img.header).set_intent("vector", (), "")
    field_path = tmp_path / "field.nii"
    field_img.to_filename(str(field_path))

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--reference",
            str(ref_path),
            "--transform",
            str(field_path),
            "--transform-type",
            "field",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    # Output adopts the reference grid shape (4,4,4), not the input (6,6,6).
    assert out.shape == (4, 4, 4)


def test_apply_warp_field_with_non_itk_format(argv, tmp_path):
    """A zero field in fsl/ants/afni format should still resample to identity
    after the format conversion in convert_warp."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    img_data = rng.random((5, 5, 5), dtype=np.float32)
    in_path = tmp_path / "img.nii"
    nib.Nifti1Image(img_data, affine).to_filename(str(in_path))

    field_path = tmp_path / "field_fsl.nii"
    nib.Nifti1Image(np.zeros((5, 5, 5, 3), dtype=np.float32), affine).to_filename(
        str(field_path)
    )

    out_path = tmp_path / "out.nii"
    argv(
        [
            "wk-apply-warp",
            "--input",
            str(in_path),
            "--transform",
            str(field_path),
            "--transform-type",
            "field",
            "--format",
            "fsl",
            "--output",
            str(out_path),
        ]
    )
    apply_warp_main()
    out = _load(str(out_path))
    np.testing.assert_allclose(out.get_fdata().squeeze(), img_data, atol=1e-3)


# ---------------------------------------------------------------------------
# wk-compute-jacobian coverage gaps: non-itk format, --frame, 5D series.
# ---------------------------------------------------------------------------


def test_compute_jacobian_field_with_ants_format(argv, tmp_path):
    """ANTs-format zero field still routes through convert_warp before the
    Jacobian computation; result is identically 1."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field_5d = np.zeros((6, 6, 6, 1, 3), dtype=np.float32)
    field_img = nib.Nifti1Image(field_5d, affine)
    cast(nib.Nifti1Header, field_img.header).set_intent("vector", (), "")
    field_path = tmp_path / "ants.nii"
    field_img.to_filename(str(field_path))

    out_path = tmp_path / "jdet.nii"
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(field_path),
            "--output",
            str(out_path),
            "--from",
            "field",
            "--from-format",
            "ants",
        ]
    )
    compute_jacobian_main()
    np.testing.assert_allclose(_load(str(out_path)).get_fdata(), 1.0, atol=1e-5)


def test_compute_jacobian_with_frame(argv, tmp_path):
    """--frame N picks the Nth field of a 5D field series."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field_5d = np.zeros((5, 5, 5, 3, 3), dtype=np.float32)
    field_img = nib.Nifti1Image(field_5d, affine)
    cast(nib.Nifti1Header, field_img.header).set_intent("vector", (), "")
    field_path = tmp_path / "field5d.nii"
    field_img.to_filename(str(field_path))

    out_path = tmp_path / "jdet.nii"
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(field_path),
            "--output",
            str(out_path),
            "--from",
            "field",
            "--frame",
            "1",
        ]
    )
    compute_jacobian_main()
    j = _load(str(out_path))
    assert j.shape == (5, 5, 5)
    np.testing.assert_allclose(j.get_fdata(), 1.0, atol=1e-5)


def test_compute_jacobian_bundled_vs_per_frame_outputs_match(argv, tmp_path):
    """Bundled (single output) and per-frame (N outputs) writes of the same
    input produce identical data."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    map_data = (rng.standard_normal((5, 5, 5, 3)) * 0.1).astype(np.float32)
    map_path = tmp_path / "maps.nii"
    nib.Nifti1Image(map_data, affine).to_filename(str(map_path))

    bundled_path = tmp_path / "jdet_bundled.nii"
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(map_path),
            "--output",
            str(bundled_path),
            "--from",
            "map",
            "--axis",
            "j",
        ]
    )
    compute_jacobian_main()
    bundled = _load(str(bundled_path)).get_fdata()
    assert bundled.shape == (5, 5, 5, 3)

    per_frame_paths = [tmp_path / f"jdet_{i}.nii" for i in range(3)]
    argv(
        [
            "wk-compute-jacobian",
            "--input",
            str(map_path),
            "--output",
            *[str(p) for p in per_frame_paths],
            "--from",
            "map",
            "--axis",
            "j",
        ]
    )
    compute_jacobian_main()
    for i, p in enumerate(per_frame_paths):
        np.testing.assert_allclose(_load(str(p)).get_fdata(), bundled[..., i], atol=0)


# ---------------------------------------------------------------------------
# wk-convert-warp coverage gaps: invert + format conversion + axis combo,
# multi-frame field series invert.
# ---------------------------------------------------------------------------


def test_convert_warp_invert_field_with_format_conversion(argv, tmp_path):
    """Single-frame field input in fsl format + --invert. Routes through
    convert_warp(fsl→itk) before inversion. Zero field inverts to zero."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field = np.zeros((5, 5, 5, 3), dtype=np.float32)
    in_path = tmp_path / "field_fsl.nii"
    nib.Nifti1Image(field, affine).to_filename(str(in_path))

    out_path = tmp_path / "field_inv.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--from",
            "field",
            "--from-format",
            "fsl",
            "--to-format",
            "itk",
            "--invert",
        ]
    )
    convert_warp_main()
    out = _load(str(out_path))
    np.testing.assert_allclose(out.get_fdata(), 0.0, atol=1e-5)


def test_convert_warp_multi_frame_field_invert_routes_through_map_inverter(
    argv, tmp_path
):
    """Multi-frame field input + --invert: route extracts per-frame channel
    along --axis, runs the 1D map inverter, then promotes maps back to fields
    for the field output (default --to matches --from). Zero in, zero out."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    fields = []
    for i in range(3):
        path = tmp_path / f"f{i}.nii"
        nib.Nifti1Image(np.zeros((5, 5, 5, 3), dtype=np.float32), affine).to_filename(
            str(path)
        )
        fields.append(str(path))

    out_path = tmp_path / "inv.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            *fields,
            "--output",
            str(out_path),
            "--from",
            "field",
            "--invert",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()
    loaded = _load(str(out_path))
    # Output is a 5D field series (3 frames, 3 channels).
    assert loaded.shape == (5, 5, 5, 3, 3)
    assert loaded.header.get_intent()[0] == "vector"
    np.testing.assert_allclose(loaded.get_fdata(), 0.0, atol=1e-5)


def test_convert_warp_multi_frame_field_invert_to_map_output(argv, tmp_path):
    """Multi-frame field input + --invert + --to=map: maps come out of the
    inverter (1-channel along --axis) and stay as a map series. Zero in, zero
    out, and the map series header has no vector intent."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    fields = []
    for i in range(3):
        path = tmp_path / f"f{i}.nii"
        nib.Nifti1Image(np.zeros((5, 5, 5, 3), dtype=np.float32), affine).to_filename(
            str(path)
        )
        fields.append(str(path))

    out_path = tmp_path / "inv_maps.nii"
    argv(
        [
            "wk-convert-warp",
            "--input",
            *fields,
            "--output",
            str(out_path),
            "--from",
            "field",
            "--to",
            "map",
            "--invert",
            "--axis",
            "j",
        ]
    )
    convert_warp_main()
    loaded = _load(str(out_path))
    # Map series: (X, Y, Z, T) without channel dim; intent is not 'vector'.
    assert loaded.shape == (5, 5, 5, 3)
    assert loaded.header.get_intent()[0] != "vector"
    np.testing.assert_allclose(loaded.get_fdata(), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# wk-convert-fieldmap coverage gaps: multi-file fieldmap input, 5D field input.
# ---------------------------------------------------------------------------


def test_convert_fieldmap_multi_file_input(argv, tmp_path):
    """Multiple --input fieldmap files are flattened into a frame series."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    fmaps = []
    for i in range(3):
        path = tmp_path / f"fmap_{i}.nii"
        nib.Nifti1Image(
            np.full((4, 4, 4), 10.0 * (i + 1), dtype=np.float32), affine
        ).to_filename(str(path))
        fmaps.append(str(path))

    out_path = tmp_path / "dmap.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            *fmaps,
            "--output",
            str(out_path),
            "--from",
            "fieldmap",
            "--to",
            "map",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "k",  # voxel z=2, no LPS flip → disp = +fmap*0.05*2
        ]
    )
    convert_fieldmap_main()
    bundled = _load(str(out_path)).get_fdata()
    # Three 1-channel maps stacked into a 4D series; frame i has fmap=10*(i+1).
    assert bundled.shape == (4, 4, 4, 3)
    for i in range(3):
        np.testing.assert_allclose(
            bundled[..., i], 10.0 * (i + 1) * 0.05 * 2.0, atol=1e-5
        )


def test_convert_fieldmap_5d_field_input(argv, tmp_path):
    """5D (X,Y,Z,T,3) field series input gets split into per-frame fields
    before unit conversion."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field_5d = np.zeros((4, 4, 4, 2, 3), dtype=np.float32)
    field_5d[..., 0, 1] = 1.0  # frame 0: 1mm displacement on j channel
    field_5d[..., 1, 1] = 2.0  # frame 1: 2mm displacement on j channel
    field_path = tmp_path / "field5d.nii"
    img = nib.Nifti1Image(field_5d, affine)
    cast(nib.Nifti1Header, img.header).set_intent("vector", (), "")
    img.to_filename(str(field_path))

    out_path = tmp_path / "fmap.nii"
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            str(field_path),
            "--output",
            str(out_path),
            "--from",
            "field",
            "--to",
            "fieldmap",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
        ]
    )
    convert_fieldmap_main()
    fmap = _load(str(out_path)).get_fdata()
    # Two frames bundled into 4D scalar series.
    assert fmap.shape == (4, 4, 4, 2)
    # |fmap[i]| / |dmap[i]| = 1 / (trt * voxel) = 1 / 0.1 = 10 Hz/mm
    # frame 0: 1mm → 10 Hz; frame 1: 2mm → 20 Hz (sign depends on PE convention)
    np.testing.assert_allclose(np.abs(fmap[..., 0]), 10.0, atol=1e-4)
    np.testing.assert_allclose(np.abs(fmap[..., 1]), 20.0, atol=1e-4)


# ---------------------------------------------------------------------------
# input-shape error messages from _warp_io
# ---------------------------------------------------------------------------


def test_warp_io_map_input_5d_errors(argv, capsys, tmp_path):
    """--from=map with a 5D file errors out (maps are 3D or 4D)."""
    bad = _write_nifti(tmp_path / "bad.nii", (4, 4, 4, 1, 3))
    argv(
        [
            "wk-convert-warp",
            "--input",
            bad,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "map",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "map input must be 3D or 4D" in err


def test_warp_io_field_input_4d_wrong_channel_errors(argv, capsys, tmp_path):
    """--from=field with a 4D file whose last dim isn't 3 errors out."""
    bad = _write_nifti(tmp_path / "bad.nii", (4, 4, 4, 7))
    argv(
        [
            "wk-convert-warp",
            "--input",
            bad,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "field",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "field input must be 4D" in err


def test_apply_warp_invalid_field_shape_errors(argv, capsys, tmp_path):
    """A 4D --transform whose last dim != 3 with --transform-type=field is
    rejected (fields must be 4D X,Y,Z,3 or 5D X,Y,Z,T,3)."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))
    bad_field = _write_nifti(tmp_path / "bad.nii", (4, 4, 4, 7))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            bad_field,
            "--transform-type",
            "field",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "displacement field must be 4D" in err


def test_convert_fieldmap_frame_out_of_range_errors(argv, capsys, tmp_path):
    """wk-convert-fieldmap --frame N where N >= number of frames is rejected."""
    fmap = _write_nifti(tmp_path / "fmap.nii", (4, 4, 4, 3))
    argv(
        [
            "wk-convert-fieldmap",
            "--input",
            fmap,
            "--output",
            str(tmp_path / "out.nii"),
            "--from",
            "fieldmap",
            "--to",
            "map",
            "--total-readout-time",
            "0.05",
            "--phase-encoding-direction",
            "j",
            "--frame",
            "10",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        convert_fieldmap_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "out of range" in err


def test_unwrap_phase_noiseframes_strips_trailing_volumes(
    argv, test_data_paths, tmp_path
):
    """--noiseframes N drops the last N volumes from each input. Run on the
    bundled fixture (15 volumes) with -f 1 and assert the unwrapped output
    has 14 volumes."""
    out_prefix = tmp_path / "noise"
    argv(
        [
            "wk-unwrap-phase",
            "--magnitude",
            *test_data_paths["mag"],
            "--phase",
            *test_data_paths["phase"],
            "--metadata",
            *test_data_paths["metadata"],
            "--out-prefix",
            str(out_prefix),
            "-n",
            "1",
            "-f",
            "1",  # drop last frame
        ]
    )
    unwrap_phase_main()
    unwrapped = sorted(tmp_path.glob("noise_unwrapped_echo-*.nii"))
    assert len(unwrapped) == 3
    for u in unwrapped:
        assert _load(str(u)).shape == (64, 64, 40, 14)  # was 15, minus 1


def test_bundle_frames_to_field_series_squeezes_5d_singleton_input(tmp_path):
    """When per-frame fields are themselves 5D (X,Y,Z,1,3), the bundler must
    squeeze the singleton 4th axis before stacking — otherwise the result
    would be 6D."""
    from warpkit.scripts._warp_io import bundle_frames_to_field_series

    frames = []
    for _ in range(2):
        data = np.zeros((4, 4, 4, 1, 3), dtype=np.float32)
        frames.append(nib.Nifti1Image(data, np.eye(4)))
    bundled = bundle_frames_to_field_series(frames)
    assert bundled.shape == (4, 4, 4, 2, 3)


def test_apply_warp_with_5d_field_with_unsupported_last_axis_errors(
    argv, capsys, tmp_path
):
    """A 5D --transform whose last dim != 3 with --transform-type=field is
    rejected. Covers the 5D-but-not-3-channel branch in the field validator."""
    inp = _write_nifti(tmp_path / "in.nii", (4, 4, 4))
    bad = _write_nifti(tmp_path / "bad5d.nii", (4, 4, 4, 1, 7))
    argv(
        [
            "wk-apply-warp",
            "--input",
            inp,
            "--transform",
            bad,
            "--transform-type",
            "field",
            "--output",
            str(tmp_path / "out.nii"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        apply_warp_main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "displacement field must be 4D" in err
