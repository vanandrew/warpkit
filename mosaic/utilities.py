import numpy as np


def rescale_phase(data: np.ndarray, min: int = -4096, max: int = 4096) -> np.ndarray:
    """Rescale phase data to [-pi, pi]

    Rescales data to [-pi, pi] using the specified min and max inputs.

    Parameters
    ----------
    data : np.ndarray
        phase data to be rescaled
    min : int, optional
        min value that should be mapped to -pi, by default -4096
    max : int, optional
        max value that should be mapped to pi, by default 4096

    Returns
    -------
    np.ndarray
        rescaled phase data
    """
    return (data - min) / (max - min) * 2 * np.pi - np.pi
