import numpy as np
import numpy.typing as npt
import nibabel as nib
import logging
from typing import cast, List, Union


def fit_motion_model(
    fieldmap_data: Union[List[npt.NDArray], List[nib.Nifti1Image], npt.NDArray, nib.Nifti1Image],
    motion_params: Union[List[npt.NDArray], npt.NDArray],
) -> npt.NDArray:
    """Fits motion model to field map data.

    This fits a first order linear motion model to the MEDIC field map data. It is designed
    to operate over an entire session, fitting a separate intercept for each run (model differences
    in scanner shimming between runs).

    Parameters
    ----------
    fieldmap_data : Union[List[npt.ArrayLike], List[nib.Nifti1Image], npt.NDArray, nib.Nifti1Image]
        List of field map data for each run in the session. Each item in the list
        should represent a single run of MEDIC field maps in a numpy array or nib.Nifti1Image.
    motion_params : List[npt.ArrayLike]
        List of motion parameters for each run in the session. Each item in the list
        should represent a rigid-body (6 parameter) motion correction matrix where each
        row is a timepoint and each column is a parameter (dx, dy, dz, rx, ry, rz).

    Returns
    -------
    np.ndarray
        Array of weights for each voxel in the field map data. The last dimension is the index
        of the weight (intercepts of each run, dx, dy, dz, rx, ry, rz).
    """
    if not isinstance(fieldmap_data, list):
        fieldmap_data = cast(Union[List[npt.NDArray], List[nib.Nifti1Image]], [fieldmap_data])

    if not isinstance(motion_params, list):
        motion_params = cast(List[npt.NDArray], [motion_params])

    # check type and convert appropriately
    if isinstance(fieldmap_data[0], nib.Nifti1Image):
        fieldmap_data = [cast(nib.Nifti1Image, f).get_fdata() for f in fieldmap_data]
    fieldmap_data = cast(List[np.ndarray], fieldmap_data)

    # get number of runs
    n_runs = len(fieldmap_data)

    # check against motion params
    if n_runs != len(motion_params):
        raise ValueError("Number of runs in fieldmap_data and motion_params do not match.")

    # get spatial dimensions of field map data
    shape = fieldmap_data[0].shape[:-1]

    # get the total number of timepoints for each run
    n_timepoints = [p.shape[0] for p in motion_params]

    # construct response matrix (field map data)
    logging.info("Constructing response matrix...")
    Y = np.concatenate([f.reshape(-1, f.shape[-1]).T for f in fieldmap_data], axis=0)

    # construct design matrix (motion parameters)
    logging.info("Constructing design matrix...")
    X = np.concatenate(motion_params, axis=0)
    # construct intercept columns
    intercepts = np.zeros((X.shape[0], n_runs))
    n_start = 0
    for i, n in enumerate(n_timepoints):
        intercepts[n_start:n, i] = 1
        n_start += n
    # concatenate intercepts to design matrix
    X = np.concatenate([intercepts, X], axis=1)

    # fit the model
    logging.info("Fitting motion model...")
    W, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    logging.info(f"residuals: {residuals}")

    # reshape the weights
    W = W.T.reshape(*shape, -1)
    return W


def apply_motion_model(weights: npt.NDArray, motion_params: npt.NDArray, run_idx: int = 0) -> npt.NDArray:
    """Generates fieldmaps from fitted motion model.

    This function applies a fitted motion model to a given set of motion parameters. It is
    used in conjunction with the `fit_motion_model` function.

    Parameters
    ----------
    weights : npt.NDArray
        Array of weights for each voxel in the field map data. The last dimension is the index
        of the weight (intercepts of each run, dx, dy, dz, rx, ry, rz).
    motion_params : npt.NDArray
        Array of motion parameters for each frame. Each row is a timepoint
        and each column is a parameter (dx, dy, dz, rx, ry, rz).
    run_idx : int
        Index of the run to output (By default first run)

    Returns
    -------
    npt.NDArray
        Array of modeled field maps.
    """
    # check if run_idx is valid
    if run_idx >= weights.shape[-1]:
        raise ValueError(f"run_idx ({run_idx}) is out of range.")

    # get the weights for the specified run
    weights_run = weights[..., run_idx][..., np.newaxis]

    # get the last six columns of the weights
    weights_motion = weights[..., -6:]

    # concatenate weights
    weights = np.concatenate([weights_run, weights_motion], axis=-1)

    # add intercept to motion parameters
    motion_params = np.concatenate([np.ones((motion_params.shape[0], 1)), motion_params], axis=1)

    # apply weights to motion parameters to get field maps
    fieldmaps = np.dot(weights, motion_params.T)

    # return the field maps
    return fieldmaps
