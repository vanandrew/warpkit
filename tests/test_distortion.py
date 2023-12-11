import numpy as np
from warpkit.distortion import medic
from . import bids_test_data


def test_medic(bids_test_data):
    # this just tests for if the overall medic pipeline errors out
    # should add some assertion checks in the future
    _, _, fmap = medic(
        **bids_test_data,
        n_cpus=1,
    )
    # compute correlations between first frame and all other frames
    # should be a high correlation > 0.98
    data = fmap.get_fdata()
    frame1 = data[..., 0]
    for i in range(1, data.shape[-1]):
        framei = data[..., i]
        corr = np.corrcoef(frame1.ravel(), framei.ravel())[0, 1]
        print(corr)
        assert corr > 0.98, f"Correlation between frame 1 and frame {i} is only {corr} < 0.98"
