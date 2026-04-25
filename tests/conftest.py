from json import load
from pathlib import Path

from nibabel.nifti1 import Nifti1Image
from pytest import fixture

# get this directory
THISDIR = Path(__file__).parent


# fixture for test data
@fixture(scope="session")
def test_data():
    # get mag and phase data
    mag = Path(THISDIR, "data", "test_data").glob("*mag*.nii.gz")
    phase = Path(THISDIR, "data", "test_data").glob("*phase*.nii.gz")
    sidecar = Path(THISDIR, "data", "test_data").glob("*mag*.json")
    metadata = []
    for s in sidecar:
        with s.open() as f:
            metadata.append(load(f))
    return {
        "phase": [Nifti1Image.load(p) for p in phase],
        "mag": [Nifti1Image.load(m) for m in mag],
        "TEs": [m["EchoTime"] * 1000 for m in metadata],
        "total_readout_time": metadata[0]["TotalReadoutTime"],
        "phase_encoding_direction": metadata[0]["PhaseEncodingDirection"],
    }
