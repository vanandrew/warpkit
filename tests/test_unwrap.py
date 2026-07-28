import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from warpkit import unwrap as unwrap_mod
from warpkit.unwrap import (
    _branch_intercept_step,
    _select_branch,
    compute_field_maps,
    compute_offset,
    reject_outliers,
)

# ---------------------------------------------------------------------------
# reject_outliers: median + MAD, threshold m=2.0
# ---------------------------------------------------------------------------


def test_reject_outliers_keeps_inliers():
    """Tightly clustered data (`s = d/mdev` < 2) all survives the m=2.0 cut."""
    data = np.array([10.0, 10.1, 10.0, 10.1, 10.0])
    out = reject_outliers(data)
    assert_allclose(np.sort(out), np.sort(data))


def test_reject_outliers_drops_far_value():
    data = np.array([10.0, 10.1, 10.0, 10.1, 10.0, 50.0])
    out = reject_outliers(data)
    assert 50.0 not in out
    assert_allclose(np.sort(out), np.array([10.0, 10.0, 10.0, 10.1, 10.1]))


def test_reject_outliers_zero_mad_returns_all():
    """If MAD is 0 (all values equal), the function falls back to keeping
    everything (s = zeros, so all `s < m` is True)."""
    data = np.array([5.0, 5.0, 5.0, 5.0])
    out = reject_outliers(data)
    assert_allclose(out, data)


# ---------------------------------------------------------------------------
# compute_offset: regression-based 2π offset estimation
# ---------------------------------------------------------------------------


def test_compute_offset_zero_for_perfect_fit():
    """If the unwrapped data already lies on the regression line, the
    predicted phase exactly matches the observed phase and the integer 2π
    offset is zero."""
    n_echos = 4
    n_voxels = 50
    tes = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    coefs = np.linspace(0.1, 0.5, n_voxels, dtype=np.float32)
    x_mat = tes[:, np.newaxis]  # (n_echos, 1)
    # perfect data: y = coefs * te
    y_mat = (x_mat * coefs[np.newaxis, :]).astype(np.float32)
    w_mat = np.ones((n_echos, n_voxels), dtype=np.float32)
    # for echo_ind=2, prediction must match → mode of int_map is 0.
    assert compute_offset(2, w_mat, x_mat, y_mat) == 0


def test_compute_offset_recovers_known_2pi_shift():
    """Add a 2π wrap to the target echo and verify compute_offset returns -1
    (so the caller adds +2π to the wrapped echo to reverse it)."""
    n_echos = 4
    n_voxels = 50
    tes = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    coefs = np.linspace(0.1, 0.5, n_voxels, dtype=np.float32)
    x_mat = tes[:, np.newaxis]
    y_mat = (x_mat * coefs[np.newaxis, :]).astype(np.float32)
    # shift echo 2 up by 2π for every voxel
    y_mat[2, :] += 2 * np.pi
    w_mat = np.ones((n_echos, n_voxels), dtype=np.float32)
    # mode of round((y_pred - y_observed) / 2π) over all voxels = -1
    assert compute_offset(2, w_mat, x_mat, y_mat) == -1


# ---------------------------------------------------------------------------
# Romeo bindings — light smoke test that the lowercase free-function names
# exist at module level (the heavy property tests live in test_romeo).
# ---------------------------------------------------------------------------


def test_romeo_lowercase_bindings_exist():
    import warpkit.warpkit_cpp as cpp

    # Python-visible names are lowercase free functions on the module;
    # there is no Romeo class wrapper, and the original uppercase names
    # should not appear at module level either.
    assert hasattr(cpp, "romeo_unwrap3d")
    assert hasattr(cpp, "romeo_unwrap4d")
    assert hasattr(cpp, "romeo_voxelquality")
    assert hasattr(cpp, "calculate_weights")
    assert not hasattr(cpp, "Romeo")
    assert not hasattr(cpp, "romeo_unwrap3D")
    assert not hasattr(cpp, "romeo_unwrap4D")


def _make_field_inputs(spatial=(4, 4, 4), n_frames=2, n_echoes=3):
    """Build minimal valid ``compute_field_maps`` inputs (unwrapped per-echo
    4D images + per-frame masks). Caller can swap the masks to break the
    expected shape."""
    affine = np.eye(4)
    unwrapped = [
        nib.Nifti1Image(
            np.zeros((*spatial, n_frames), dtype=np.float32),
            affine,
        )
        for _ in range(n_echoes)
    ]
    mag = [
        nib.Nifti1Image(
            np.ones((*spatial, n_frames), dtype=np.float32),
            affine,
        )
        for _ in range(n_echoes)
    ]
    masks = nib.Nifti1Image(
        np.ones((*spatial, n_frames), dtype=np.int8),
        affine,
    )
    tes = [10.0, 20.0, 30.0]
    return unwrapped, mag, masks, tes


def test_compute_field_maps_rejects_3d_masks():
    """A masks input that's missing the time axis must fail loudly, not
    silently broadcast or crash deeper inside the SVD pass."""
    unwrapped, mag, _masks, tes = _make_field_inputs()
    bad_masks = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int8), np.eye(4))
    with pytest.raises(ValueError, match="masks must have shape"):
        compute_field_maps(unwrapped, bad_masks, mag, tes)


def test_compute_field_maps_rejects_mismatched_frame_count():
    """masks frame count must match the unwrapped time dimension."""
    unwrapped, mag, _masks, tes = _make_field_inputs(n_frames=2)
    bad_masks = nib.Nifti1Image(np.ones((4, 4, 4, 5), dtype=np.int8), np.eye(4))
    with pytest.raises(ValueError, match="masks must have shape"):
        compute_field_maps(unwrapped, bad_masks, mag, tes)


def test_compute_field_maps_rejects_mismatched_spatial_shape():
    unwrapped, mag, _masks, tes = _make_field_inputs(spatial=(4, 4, 4))
    bad_masks = nib.Nifti1Image(np.ones((5, 4, 4, 2), dtype=np.int8), np.eye(4))
    with pytest.raises(ValueError, match="masks must have shape"):
        compute_field_maps(unwrapped, bad_masks, mag, tes)


# ---------------------------------------------------------------------------
# Global 2*pi branch selection (replaces the field-magnitude heuristic cascade)
# ---------------------------------------------------------------------------


def _patch_scores(monkeypatch, scores):
    """Stub _evaluate_branch so only the decision rule is under test.

    ``scores`` are intercepts in radians. The selector consults nothing else --
    no field prior, no tiebreak -- so the remaining tuple slots are unused.
    """

    def fake(n_wraps, *args, **kwargs):
        return scores[n_wraps], None, None, None

    monkeypatch.setattr(unwrap_mod, "_evaluate_branch", fake)


def _sel(te0: float = 12.0, te1: float = 28.97):
    """Call the selector with the bundled protocol's TEs by default.

    ``_evaluate_branch`` is stubbed out by ``_patch_scores`` in every test that
    uses this, so the array arguments are never read -- they just need to be the
    right type. Defaults give the bundled protocol, k = 0.7071.
    """
    vol = np.zeros((2, 2, 2), dtype=np.float32)
    flags = np.ones((2, 2, 2), dtype=bool)
    return _select_branch(
        vol,  # unwrapped_diff
        vol,  # phase0
        vol,  # phase1
        np.ones((2, 2, 2), dtype=np.float32),  # mag0
        vol,  # mag1
        np.float32(te0),
        np.float32(te1),
        flags,  # mask
        flags,  # score_mask
    )


# ---------------------------------------------------------------------------
# _branch_intercept_step: how far apart candidate branches sit, from TEs alone
# ---------------------------------------------------------------------------


def test_branch_intercept_step_matches_measured_offset():
    """The bundled protocol's wrong branches measure an intercept of 1.8402 rad.
    The analytic step must reproduce that without touching any data."""
    assert _branch_intercept_step(np.float32(12.0), np.float32(28.97)) == pytest.approx(
        1.8402, rel=1e-4
    )


def test_branch_intercept_step_matches_ds007637():
    assert _branch_intercept_step(
        np.float32(14.20), np.float32(38.93)
    ) == pytest.approx(2.6754, rel=1e-4)


@pytest.mark.parametrize(("te0", "te1"), [(10.0, 20.0), (15.0, 20.0), (20.0, 30.0)])
def test_branch_intercept_step_zero_for_integer_ratio(te0, te1):
    """te0/dTE integer -> the branch cannot move the offset at all."""
    assert _branch_intercept_step(np.float32(te0), np.float32(te1)) == pytest.approx(
        0.0
    )


def test_branch_intercept_step_rejects_nonincreasing_tes():
    assert _branch_intercept_step(np.float32(20.0), np.float32(20.0)) == 0.0


def test_branch_selector_noop_for_integer_te_ratio(monkeypatch):
    """With an integer te0/dTE the selector must not act, even on a score
    spread that would otherwise look decisive -- the branch is a no-op there,
    so any apparent difference is numerical noise."""
    _patch_scores(monkeypatch, {-1: 1.84, 0: 1.84, 1: 1.0e-7})
    best, _ = _sel(te0=10.0, te1=20.0)
    assert best == 0


def test_branch_selector_corrects_inconsistent_zero(monkeypatch):
    """N=0 carries an intercept and exactly one alternative does not: move."""
    _patch_scores(monkeypatch, {-1: 1.8402, 0: 1.8402, 1: 1.0e-7})
    best, _ = _sel()
    assert best == 1


def test_branch_selector_defers_when_two_branches_fit(monkeypatch):
    """The ds007637 case: N=0 and N=-1 are both through-origin. That is a
    genuine alias -- both explain the phase exactly -- so the selector must not
    choose between them. Deferring leaves correct_global's answer in place.

    A bare argmin on the intercepts would flip between them on numerical noise
    and inject a full wrap of field into the time series.
    """
    _patch_scores(monkeypatch, {-1: 1.2e-03, 0: 1.9e-03, 1: 0.9324})
    best, _ = _sel()
    assert best == 0


def test_branch_selector_defers_even_when_zero_looks_wrong(monkeypatch):
    """N=0 carries an intercept but *two* alternatives fit. Neither is
    preferable on the evidence, so change nothing rather than pick one.

    This is the case that used to consult the field prior. Measurement showed
    the prior was right about as often as it was wrong, so it was removed.
    """
    _patch_scores(monkeypatch, {-1: 1.0e-7, 0: 1.8402, 1: 2.0e-7})
    best, _ = _sel()
    assert best == 0


def test_branch_selector_noop_when_all_candidates_tie(monkeypatch):
    """Integer TE0/dTE makes the branch a no-op; scores are all ~equal."""
    _patch_scores(monkeypatch, {-1: 1.1e-8, 0: 2.0e-8, 1: 1.9e-8})
    best, _ = _sel()
    assert best == 0


def test_branch_selector_noop_when_nothing_fits(monkeypatch):
    """Degenerate/failed fit: every candidate is bad, so change nothing."""
    _patch_scores(monkeypatch, {-1: 0.0, 0: 0.0, 1: 0.0})
    best, _ = _sel()
    assert best == 0

    _patch_scores(monkeypatch, {-1: 1.71, 0: 1.80, 1: 1.74})
    best, _ = _sel()
    assert best == 0


def test_branch_selector_recovers_injected_wrap(test_data):
    """End-to-end: shift the unwrapped difference by a known number of wraps and
    confirm the selector undoes it, landing on the same phase offset."""
    from warpkit.utilities import create_brain_mask, rescale_phase
    from warpkit.warpkit_cpp import romeo_unwrap3d

    phase, mag, tes = test_data["phase"], test_data["mag"], test_data["tes"]
    tes = np.asarray(tes, dtype=np.float32)
    raw = np.stack([p.dataobj[..., 0] for p in phase], axis=-1)
    mn = min(float(np.asarray(p.dataobj[..., 0]).min()) for p in phase)
    mx = max(float(np.asarray(p.dataobj[..., 0]).max()) for p in phase)
    ph = rescale_phase(raw, min=mn, max=mx).astype(np.float32)
    mg = np.stack([m.dataobj[..., 0] for m in mag], axis=-1).astype(np.float32)

    mag0, mag1 = mg[..., 0], mg[..., 1]
    phase0, phase1 = ph[..., 0], ph[..., 1]
    mask = create_brain_mask(mag0, 3)
    score_mask = create_brain_mask(mag0, -2)

    signal_diff = mag0 * mag1 * np.exp(1j * (phase1 - phase0))
    unwrapped_diff = romeo_unwrap3d(
        phase=np.angle(signal_diff).astype(np.float32),
        weights="romeo",
        mag=np.abs(signal_diff).astype(np.float32),
        mask=mask,
        correct_global=True,
    )

    for injected in (-1, 0, 1):
        best, scores = _select_branch(
            unwrapped_diff + 2 * np.pi * injected,
            phase0,
            phase1,
            mag0,
            mag1,
            tes[0],
            tes[1],
            mask,
            score_mask,
        )
        assert best == -injected, f"injected {injected}, got {best} (scores={scores})"


def test_romeo_unwrap3d_rejects_unknown_weight_preset():
    """`weights` is a preset name string; only "romeo" is supported."""
    from warpkit.warpkit_cpp import romeo_unwrap3d

    phase = np.zeros((3, 3, 3), dtype=np.float32)
    mag = np.ones_like(phase)
    mask = np.ones(phase.shape, dtype=bool)
    with pytest.raises(Exception, match='only the "romeo" weight preset'):
        romeo_unwrap3d(phase, "ramen", mag, mask)
