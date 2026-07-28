from json import load
from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from pytest import fixture

# get this directory
THISDIR = Path(__file__).parent
TEST_DATA_DIR = THISDIR / "data" / "test_data"
BRANCH_FLIP_DIR = THISDIR / "data" / "branch_flip"


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
def branch_flip_data():
    """Two frames from `ds006131` sub-20828 run-01 that expose a real
    ``correct_global`` branch flip.

    That subject's global field sits at 17.7 Hz against a 20.22 Hz half-wrap
    (TEs 14.2/38.93/63.66 ms), so ROMEO's median of rounded wrap counts is on a
    knife edge and tips over on individual frames. Frame 0 here is the original
    frame 43 and is healthy; frame 1 is the original frame 44, where the
    dual-echo field flips a full 40.44 Hz wrap.

    Cropped to the brain bounding box, which preserves the behaviour exactly.
    It is *not* decimated: resampling changes which voxels vote in the ballot
    and tips it the other way, which would destroy the very thing under test.
    """
    mag = sorted(BRANCH_FLIP_DIR.glob("*part-mag*.nii.gz"))
    phase = sorted(BRANCH_FLIP_DIR.glob("*part-phase*.nii.gz"))
    sidecar = sorted(BRANCH_FLIP_DIR.glob("*part-mag*.json"))
    metadata = []
    for s in sidecar:
        with s.open() as f:
            metadata.append(load(f))
    return {
        "phase": [cast(Nifti1Image, nib.load(str(p))) for p in phase],
        "mag": [cast(Nifti1Image, nib.load(str(m))) for m in mag],
        "tes": [m["EchoTime"] * 1000 for m in metadata],
        # frame index -> branch the selector must choose
        "expected_branch": {0: 0, 1: 1},
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
