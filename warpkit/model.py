import numpy as np
import numpy.typing as npt


def weighted_regression(
    x: npt.NDArray, y: npt.NDArray, w: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Single parameter weighted regression

    This function does a single parameter weighted regression using elementwise matrix operations.

    Parameters
    ----------
    x : npt.NDArray
        Design Matrix, rows are each echo, columns are voxels. If column is size 1, this array will be broadcasted
        across all voxels
    y : npt.NDArray
        Ordinate Matrix, rows are each echo, columns are voxels. (This is meant for phase values)
    w : npt.NDArray
        Weight Matrix (usually the magnitude image) echos x voxels

    Returns
    -------
    npt.NDArray
        model weights
    npt.NDArray
        residuals
    """
    # compute weighted x and y
    wy = w * y
    wx = w * x

    # now compute the weighted squared magnitude of the x
    sq_wx = np.sum(wx**2, axis=0, keepdims=True)

    # get the inverse x
    inv_wx = np.zeros(wx.shape)
    np.divide(wx, sq_wx, out=inv_wx, where=(sq_wx != 0))

    # compute the weights
    model_weights = (inv_wx * wy).sum(axis=0)

    # now compute residuals (on unweighted data)
    residuals = np.sum((y - model_weights * x) ** 2, axis=0)

    # return model weights and residuals
    return model_weights, residuals
