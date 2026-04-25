import numpy as np
from warpkit.model import weighted_regression


def test_weighted_regression():
    # compare weighted regression implementation to numpy
    size = 10
    x = np.random.rand(size, 1)
    y = np.random.rand(size, size)
    w1 = np.ones((size, size))
    w2 = 2 * w1

    # compute regression using lstsq (no weights)
    np_weights, np_residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)

    # compute using weighted_regression
    weights, residuals = weighted_regression(x, y, w1)

    # these should be the same
    assert np.all(np.isclose(np_weights, weights))
    assert np.all(np.isclose(np_residuals, residuals))

    # run with constant weight
    weights2, residuals2 = weighted_regression(x, y, w2)

    # these should also be the same
    assert np.all(np.isclose(weights, weights2))
    assert np.all(np.isclose(residuals, residuals2))
