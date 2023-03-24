from warpkit.distortion import *
from . import *


def test_medic(bids_test_data):
    # this just tests for if the overall medic pipeline errors out
    # should add some assertion checks in the future
    fmap_native, dmap, fmap = medic(
        **bids_test_data,
    )
    # fmap_native.to_filename("fmap_native.nii.gz")
    # dmap.to_filename("dmap.nii.gz")
    # fmap.to_filename("fmap.nii.gz")
