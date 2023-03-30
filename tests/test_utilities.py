from warpkit.utilities import *
from numpy.testing import assert_allclose
import numpy as np
from . import *


def test_corr2_coeff():
    # Test case 1: identical arrays should have correlation of 1.0
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    expected_result = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_allclose(corr2_coeff(A, B), expected_result)

    # Test case 2: negatively correlated columns
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    B = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]).T
    expected_result = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
    assert_allclose(corr2_coeff(A, B), expected_result)

    # Test case 3: known correlations
    size = 10
    A = np.random.rand(size, size)
    B = np.random.rand(size, 1)
    C = np.zeros(size)
    for i in range(size):
        C[i] = np.corrcoef(B[:, 0], A[:, i])[0, 1]
    assert_allclose(corr2_coeff(B, A).ravel(), C)
