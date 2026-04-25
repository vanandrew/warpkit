import os

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0, "unknown", "")

if os.environ.get("WARPKIT_DEV", False) == "1":
    from warnings import warn

    warn(
        "WARPKIT_DEV is set, trying to load from repo root build directory instead...",
        stacklevel=2,
    )
    import sys
    from pathlib import Path

    BUILD_DIR = str(Path(__file__).parent.parent / "build")
    sys.path.append(BUILD_DIR)
    from warpkit_cpp import *  # type: ignore  # noqa: F403

    warn(
        "Successfully loaded warpkit_cpp from repo root build directory!", stacklevel=2
    )
else:
    from .warpkit_cpp import *  # type: ignore  # noqa: F403
