import numpy as np
import pytest
from numpy.testing import assert_allclose
from warpkit.model import weighted_regression


def test_weighted_regression_matches_numpy_lstsq():
    """Constant weights make weighted_regression equivalent to numpy.linalg.lstsq."""
    rng = np.random.default_rng(0)
    size = 10
    x = rng.random((size, 1))
    y = rng.random((size, size))
    w1 = np.ones((size, size))
    w2 = 2 * w1

    np_weights, np_residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)
    weights, residuals = weighted_regression(x, y, w1)

    # lstsq returns weights as (1, n_voxels); weighted_regression as (n_voxels,).
    assert_allclose(np_weights.ravel(), weights)
    assert_allclose(np_residuals, residuals)

    # constant weight scaling doesn't change the solution.
    weights2, residuals2 = weighted_regression(x, y, w2)
    assert_allclose(weights, weights2)
    assert_allclose(residuals, residuals2)


def test_weighted_regression_recovers_known_slope():
    """Synthetic y = m * x with a known m → solver returns m, residuals ~ 0."""
    n_echos = 5
    n_voxels = 8
    x = np.linspace(1.0, 5.0, n_echos)[:, None]
    true_slope = np.linspace(0.5, 2.0, n_voxels)
    y = x * true_slope[None, :]
    w = np.ones((n_echos, n_voxels))

    weights, residuals = weighted_regression(x, y, w)
    assert_allclose(weights, true_slope, atol=1e-10)
    assert_allclose(residuals, 0.0, atol=1e-10)


def test_weighted_regression_handles_zero_weight_column():
    """A column where all weights are 0 short-circuits via the
    ``where=(sq_wx != 0)`` guard so we get model_weight=0, no NaNs."""
    n_echos = 4
    n_voxels = 3
    x = np.linspace(1.0, 4.0, n_echos)[:, None]
    y = np.ones((n_echos, n_voxels))
    w = np.ones((n_echos, n_voxels))
    w[:, 1] = 0.0  # zero out the middle column's weights

    weights, _ = weighted_regression(x, y, w)
    assert weights[1] == pytest.approx(0.0)
    assert not np.isnan(weights).any()
