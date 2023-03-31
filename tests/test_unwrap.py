from warpkit.unwrap import *
from . import *
import numpy as np

def test_compute_offset(bids_test_data):
    phase = bids_test_data["phase"]
    mag = bids_test_data["mag"]
    TEs = bids_test_data["TEs"]
    
    TEs = cast(npt.NDArray[np.float64], np.array(TEs))
    frames = list(range(phase[0].shape[-1]))
    residual_unwrapped = np.zeros((*phase[0].shape[:3], len(TEs), len(frames)))
    regular_unwrapped = np.zeros((*phase[0].shape[:3], len(TEs), len(frames)))
    mask = SimpleNamespace()
    mask.dataobj = np.ones((1, 1, 1, phase[0].shape[-1]))

    residual_unwrapped, _ = start_unwrap_process(residual_unwrapped, phase, mag, TEs, mask, frames, automask=True, automask_dilation=3, correct_global=True, n_cpus=1, residual_offset=True)
    regular_unwrapped, _ = start_unwrap_process(regular_unwrapped, phase, mag, TEs, mask, frames, automask=True, automask_dilation=3, correct_global=True, n_cpus=1, residual_offset=False)

    assert(np.array_equal(residual_unwrapped, regular_unwrapped)) == True

def test_temporal_consistency(bids_test_data):
    # test that we can compute the temporal consistency between two images
    phase = bids_test_data["phase"]
    mag = bids_test_data["mag"]
    TEs = bids_test_data["TEs"]
    
    # compute fieldmap multithreaded
    threaded_fieldmap = unwrap_and_compute_field_maps(phase, mag, TEs, frames=None, n_cpus=8, debug=False)

    # compute fieldmap single threaded
    serial_fieldmap = unwrap_and_compute_field_maps(phase, mag, TEs, frames=None, n_cpus=1, debug=False)

    assert(np.array_equal(threaded_fieldmap.get_fdata(), serial_fieldmap.get_fdata())) == True
