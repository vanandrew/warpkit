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
    import argparse

    from warpkit.scripts._warp_io import write_output

    frames = [_vector_intent_frame() for _ in range(2)]
    out_paths = [str(tmp_path / "f1.nii"), str(tmp_path / "f2.nii")]

    write_output(frames, out_paths, "map", argparse.ArgumentParser())

    for p in out_paths:
        loaded = _load(p)
        assert loaded.header.get_intent()[0] != "vector"
