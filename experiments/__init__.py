from pytest import fixture
from bids import BIDSLayout
import nibabel as nib


@fixture
def test_data():
    layout = BIDSLayout("/home/vanandrew/Data/bidsdata")
    # get mag and phase data
    mag = layout.get(subject="MSCHD02", part="mag", datatype="func", suffix="bold", extension="nii.gz")
    phase = layout.get(subject="MSCHD02", part="phase", datatype="func", suffix="bold", extension="nii.gz")
    return {
        "phase": phase,
        "mag": mag,
        "TEs": [m.get_metadata()["EchoTime"] * 1000 for m in mag],
        "TotalReadoutTime": phase[0].get_metadata()["TotalReadoutTime"],
        "PhaseEncodingDirection": phase[0].get_metadata()["PhaseEncodingDirection"],
    }


@fixture
def test_warp():
    return nib.load("/home/vanandrew/Data/test.nii.gz")
