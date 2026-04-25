import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from warpkit.unwrap import compute_field_maps, compute_offset, reject_outliers

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


def test_romeo_unwrap3d_rejects_unknown_weight_preset():
    """`weights` is a preset name string; only "romeo" is supported."""
    from warpkit.warpkit_cpp import romeo_unwrap3d

    phase = np.zeros((3, 3, 3), dtype=np.float32)
    mag = np.ones_like(phase)
    mask = np.ones(phase.shape, dtype=bool)
    with pytest.raises(Exception, match='only the "romeo" weight preset'):
        romeo_unwrap3d(phase, "ramen", mag, mask)
