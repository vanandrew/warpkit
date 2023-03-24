from pytest import fixture
from bids import BIDSLayout
from pathlib import Path


# get this directory
THISDIR = Path(__file__).parent


# fixture for bids test data
@fixture(scope="session")
def bids_test_data():
    layout = BIDSLayout(THISDIR / "data" / "bidsdata")
    # get mag and phase data
    mag = layout.get(subject="MSCHD02", part="mag", datatype="func", suffix="bold", extension="nii.gz")
    phase = layout.get(subject="MSCHD02", part="phase", datatype="func", suffix="bold", extension="nii.gz")
    return {
        "phase": [p.get_image() for p in phase],
        "mag": [m.get_image() for m in mag],
        "TEs": [m.get_metadata()["EchoTime"] * 1000 for m in mag],
        "total_readout_time": phase[0].get_metadata()["TotalReadoutTime"],
        "phase_encoding_direction": phase[0].get_metadata()["PhaseEncodingDirection"],
    }
