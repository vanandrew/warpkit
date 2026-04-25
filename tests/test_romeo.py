"""
Tests for warpkit's C++ port of ROMEO (phase unwrapping + voxel quality).

Test strategy mirrors ROMEO.jl v1.0.0's own test suite:

- Literal-golden tests (`test_unwrap_1d_literals`, `test_weight_calc_literals`)
  port numeric constants directly from upstream `dsp_tests.jl` and
  `specialcases.jl`.
- Property tests (`test_unwrap3d_property`, `test_unwrap4d_property`) replay
  the `mri.jl` strategy: an unwrapped result must differ from the wrapped input
  by exact multiples of 2π inside the mask.
- Behavioral tests (`test_voxelquality_behavior`) mirror `voxelquality.jl`:
  outputs must be finite, in [0, 1], and differ across kwarg variants.

The Phase.nii + Mag.nii volumes under `tests/data/romeo/` are copied from
ROMEO.jl's `test/data/small/` at commit 9faef5bb (MIT, attribution in
include/romeo/LICENSE).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROMEO_TEST_DATA = Path(__file__).parent / "data" / "romeo"


def _load_nii(path: Path) -> np.ndarray:
    """Load a NIfTI file as float32, bypassing the slope=NaN header that
    ROMEO's bundled test data ships with."""
    img = nib.load(str(path))
    # nibabel's base type signature doesn't promise `.dataobj`; narrow to
    # Nifti1Image since these are concretely .nii files. `.get_unscaled()` is
    # on ArrayProxy but nibabel's stubs type dataobj as ArrayLike.
    assert isinstance(img, nib.Nifti1Image)
    return np.asarray(img.dataobj.get_unscaled(), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.fixture(scope="module")
def phase4d() -> np.ndarray:
    """Synthetic phase volume from ROMEO.jl, shape (51, 51, 41, 3), float32, wrapped to [-π, π]."""
    return _load_nii(ROMEO_TEST_DATA / "Phase.nii")


@pytest.fixture(scope="module")
def mag4d() -> np.ndarray:
    """Matching magnitude volume, shape (51, 51, 41, 3), float32, non-negative."""
    return _load_nii(ROMEO_TEST_DATA / "Mag.nii")


@pytest.fixture(scope="module")
def romeo():
    """Handle to the ROMEO C++ context."""
    from warpkit.warpkit_cpp import Romeo

    return Romeo()


# ---------------------------------------------------------------------------
# Literal goldens — ported from ROMEO.jl test/dsp_tests.jl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapped",
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2 + 2 * np.pi, 0.3, 0.4],
        [0.1, 0.2 - 2 * np.pi, 0.3, 0.4],
        [0.1, 0.2 - 2 * np.pi, 0.3 - 2 * np.pi, 0.4],
        [0.1 + 2 * np.pi, 0.2, 0.3, 0.4],
    ],
    ids=[
        "nowrap",
        "wrap-up-idx1",
        "wrap-down-idx1",
        "wrap-down-idx1-2",
        "wrap-up-idx0",
    ],
)
def test_unwrap_1d_literals(romeo, wrapped):
    """ROMEO.jl's unwrap!() reshapes <=2D input to 3D before running. Reproduce
    by feeding a (N, 1, 1) volume through the 3D entry point."""
    expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    phase = np.asarray(wrapped, dtype=np.float32).reshape(-1, 1, 1)
    mag = np.ones_like(phase)
    mask = np.ones(phase.shape, dtype=bool)
    unwrapped = romeo.romeo_unwrap3d(phase, "romeo", mag, mask).reshape(-1)
    np.testing.assert_allclose(unwrapped, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Weight-calculation literals — ported from ROMEO.jl test/specialcases.jl
#
# The NaN-neighbor expected value is 119, not the 157 from the vendored
# v1.0.0 test file. ROMEO.jl master hardened `phaselinearity` to return 0.5
# when its input triplet contains a NaN, and its own specialcases.jl now
# expects 119 (gated on `VERSION ≥ v"1.8"`). The C++ port adopts the master
# NaN guard because it's more robust and deterministic across platforms.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase,expected",
    [
        (
            [0.1, 0.2 + 2 * np.pi, 0.3, 0.4],
            [30, 7, 30, 0],
        ),  # phase-linearity penalty (30) at borders
        (
            [0.1, 0.2 + 2 * np.pi, 0.3, np.nan],
            [30, 119, 0, 0],
        ),  # 119 via phaselinearity NaN→0.5 guard
    ],
    ids=["linearity-border", "nan-neighbor"],
)
def test_weight_calc_literals(romeo, phase, expected):
    """Port of the weight_test() assertions from specialcases.jl."""
    phase_arr = np.asarray(phase, dtype=np.float32).reshape(-1, 1, 1)
    # (3, nx, ny, nz) uint8 — we check dim-0 edges along the length-4 x-axis.
    weights = romeo.calculate_weights(phase_arr)
    np.testing.assert_array_equal(
        weights[0, :, 0, 0], np.asarray(expected, dtype=np.uint8)
    )


# ---------------------------------------------------------------------------
# Property tests on the 4D Phase.nii / Mag.nii fixtures — ported from mri.jl
# ---------------------------------------------------------------------------


def _rem2pi_nearest(x: np.ndarray) -> np.ndarray:
    """Equivalent of Julia's rem2pi(x, RoundNearest): fold x into (-π, π]."""
    return x - 2 * np.pi * np.round(x / (2 * np.pi))


def test_unwrap3d_property(romeo, phase4d, mag4d):
    """ROMEO's 3D unwrap must differ from wrapped input only by multiples of 2π."""
    echo = 2  # index for the 3rd echo, matching Julia's `echo = 3` (1-based)
    wrapped = np.ascontiguousarray(phase4d[..., echo])
    mag = np.ascontiguousarray(mag4d[..., echo])
    mask = np.ones(wrapped.shape, dtype=bool)

    unwrapped = romeo.romeo_unwrap3d(wrapped, "romeo", mag, mask)

    assert unwrapped.shape == wrapped.shape
    assert not np.array_equal(unwrapped, wrapped), "unwrap returned the input unchanged"
    assert np.isfinite(unwrapped).all(), "unwrap produced non-finite values"
    residual = _rem2pi_nearest(unwrapped - wrapped)
    np.testing.assert_allclose(residual, 0.0, atol=1e-5)


def test_unwrap4d_property(romeo, phase4d, mag4d):
    """Same 2π-modulo invariant across every echo of the 4D multi-echo unwrap."""
    tes = np.array([4.0, 8.0, 12.0], dtype=np.float32)  # matches voxelquality.jl
    mask = np.ones(phase4d.shape[:3], dtype=bool)

    unwrapped = romeo.romeo_unwrap4d(phase4d, tes, "romeo", mag4d, mask)

    assert unwrapped.shape == phase4d.shape
    assert np.isfinite(unwrapped).all()
    for e in range(phase4d.shape[-1]):
        residual = _rem2pi_nearest(unwrapped[..., e] - phase4d[..., e])
        np.testing.assert_allclose(
            residual, 0.0, atol=1e-5, err_msg=f"echo {e} failed 2π-modulo check"
        )


# ---------------------------------------------------------------------------
# Voxel quality behavior — ported from voxelquality.jl
# ---------------------------------------------------------------------------


def test_voxelquality_behavior(romeo, phase4d, mag4d):
    """
    Mirror the three-variant comparison from voxelquality.jl. ROMEO's
    voxelquality entry point requires tes and mag, so we reuse the 4D volume
    and vary the echo time ordering to produce distinct qmaps.
    """
    tes = np.array([4.0, 8.0, 12.0], dtype=np.float32)

    qm_uniform_mag = romeo.romeo_voxelquality(phase4d, tes, np.ones_like(mag4d))
    qm_real_mag = romeo.romeo_voxelquality(phase4d, tes, mag4d)
    qm_reordered = romeo.romeo_voxelquality(phase4d, tes[::-1].copy(), mag4d)

    for qmap, label in [
        (qm_uniform_mag, "uniform-mag"),
        (qm_real_mag, "real-mag"),
        (qm_reordered, "reordered-tes"),
    ]:
        assert qmap.shape == phase4d.shape[:3], f"{label}: wrong shape {qmap.shape}"
        assert np.isfinite(qmap).all(), f"{label}: non-finite qmap values"
        assert (qmap >= 0).all() and (qmap <= 1).all(), f"{label}: qmap out of [0, 1]"

    # Julia asserts qm1 != qm2 != qm3; mirror that here.
    assert not np.array_equal(qm_uniform_mag, qm_real_mag)
    assert not np.array_equal(qm_uniform_mag, qm_reordered)
    assert not np.array_equal(qm_real_mag, qm_reordered)
