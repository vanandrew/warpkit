from json import load
from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from pytest import fixture

# get this directory
THISDIR = Path(__file__).parent
TEST_DATA_DIR = THISDIR / "data" / "test_data"


# fixture for test data
@fixture(scope="session")
def test_data():
    # get mag and phase data
    mag = sorted(TEST_DATA_DIR.glob("*mag*.nii.gz"))
    phase = sorted(TEST_DATA_DIR.glob("*phase*.nii.gz"))
    sidecar = sorted(TEST_DATA_DIR.glob("*mag*.json"))
    metadata = []
    for s in sidecar:
        with s.open() as f:
            metadata.append(load(f))
    return {
        "phase": [cast(Nifti1Image, nib.load(str(p))) for p in phase],
        "mag": [cast(Nifti1Image, nib.load(str(m))) for m in mag],
        "tes": [m["EchoTime"] * 1000 for m in metadata],
        "total_readout_time": metadata[0]["TotalReadoutTime"],
        "phase_encoding_direction": metadata[0]["PhaseEncodingDirection"],
    }


@fixture(scope="session")
def test_data_paths():
    """File paths for the bundled BIDS-style MEDIC test data, suitable for
    passing directly to a CLI."""
    return {
        "mag": [str(p) for p in sorted(TEST_DATA_DIR.glob("*mag*.nii.gz"))],
        "phase": [str(p) for p in sorted(TEST_DATA_DIR.glob("*phase*.nii.gz"))],
        "metadata": [str(p) for p in sorted(TEST_DATA_DIR.glob("*mag*.json"))],
    }
