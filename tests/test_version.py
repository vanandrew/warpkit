import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import warpkit


def test_version_falls_back_when_package_metadata_missing():
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        importlib.reload(warpkit)
        assert warpkit.__version__ == "0.0.0+unknown"
    importlib.reload(warpkit)
    assert warpkit.__version__ != "0.0.0+unknown"
