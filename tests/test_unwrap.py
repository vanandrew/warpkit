from warpkit.unwrap import *
from . import *
import numpy as np

# fixtures can be found in __init__.py
# they can be used as follows:
# def test_something(bids_test_data):
#     # do something with bids_test_data
#
# where bids_test_data is a dictionary (see __init__.py for details)


# TODO: FINISH ME
def test_compute_offset(bids_test_data):
    # test that we can compute the offset between two images
    pass
    
def test_temporal_consistency(bids_test_data):
    # test that we can compute the temporal consistency between two images
    phase = bids_test_data["phase"]
    mag = bids_test_data["mag"]
    TEs = bids_test_data["TEs"]
    total_readout_time = bids_test_data["total_readout_time"]
    phase_encoding_direction = bids_test_data["phase_encoding_direction"]
    
    # compute fieldmap multithreaded
    threaded_fieldmap = unwrap_and_compute_field_maps(phase, mag, TEs, frames=None, n_cpus=8, debug=False)

    # compute fieldmap single threaded
    serial_fieldmap = unwrap_and_compute_field_maps(phase, mag, TEs, frames=None, n_cpus=1, debug=False)

    assert(np.array_equal(threaded_fieldmap.get_fdata(), serial_fieldmap.get_fdata())) == True

