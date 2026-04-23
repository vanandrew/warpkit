import os

if os.environ.get("WARPKIT_DEV", False) == "1":
    from warnings import warn

    warn("WARPKIT_DEV is set, trying to load from repo root build directory instead...")
    import sys
    from pathlib import Path

    BUILD_DIR = str(Path(__file__).parent.parent / "build")
    sys.path.append(BUILD_DIR)
    from warpkit_cpp import *  # type: ignore  # noqa: F403

    warn("Successfully loaded warpkit_cpp from repo root build directory!")
else:
    from .warpkit_cpp import *  # type: ignore  # noqa: F403

# Transitional alias — the C++ extension now exposes `Romeo` (Julia is gone).
# warpkit/unwrap.py and the rest of the call sites still refer to
# `JuliaContext`; keep the alias until those migrate in phase 6.
try:
    JuliaContext = Romeo  # type: ignore[name-defined, has-type]  # noqa: F405, F821
except NameError:
    pass
