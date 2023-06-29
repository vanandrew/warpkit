from warpkit.unwrap import *
from . import *
import numpy as np


# def test_temporal_consistency(bids_test_data):
#     # test that we can compute the temporal consistency between two images
#     phase = bids_test_data["phase"]
#     mag = bids_test_data["mag"]
#     TEs = bids_test_data["TEs"]

#     # compute fieldmap multithreaded
#     threaded_fieldmap = unwrap_and_compute_field_maps(phase, mag, TEs, frames=None, n_cpus=8, debug=False)

#     # compute fieldmap single threaded
#     serial_fieldmap = unwrap_and_compute_field_maps(phase, mag, TEs, frames=None, n_cpus=1, debug=False)

#     assert(np.array_equal(threaded_fieldmap.get_fdata(), serial_fieldmap.get_fdata()))
