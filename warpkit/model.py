import numpy as np
import numpy.typing as npt
from typing import Tuple


def weighted_regression(X: npt.NDArray, Y: npt.NDArray, W: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Single parameter weighted regression

    This function does a single parameter weighted regression using elementwise matrix operations.

    Parameters
    ----------
    X : npt.NDArray
        Design Matrix, rows are each echo, columns are voxels. If column is size 1, this array will be broadcasted
        across all voxels
    Y : npt.NDArray
        Ordinate Matrix, rows are each echo, columns are voxels. (This is meant for phase values)
    W : npt.NDArray
        Weight Matrix (usually the magnitude image) echos x voxels

    Returns
    -------
    npt.NDArray
        model weights
    npt.NDArray
        residuals
    """
    # compute weighted X and Y
    WY = W * Y
    WX = W * X

    # now compute the weighted squared magnitude of the X
    sq_WX = np.sum(WX**2, axis=0, keepdims=True)

    # get the inverse X
    inv_WX = np.zeros(WX.shape)
    np.divide(WX, sq_WX, out=inv_WX, where=(sq_WX != 0))

    # compute the weights
    model_weights = (inv_WX * WY).sum(axis=0)

    # now compute residuals (on unweighted data)
    residuals = np.sum((Y - model_weights * X) ** 2, axis=0)

    # return model weights and residuals
    return model_weights, residuals
