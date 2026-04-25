import importlib.util

# PyInstaller's static analysis usually picks up `warpkit_cpp.cpython-*.so` via
# the import graph (`from .warpkit_cpp import *` in warpkit/__init__.py), but
# only when the resolved warpkit package directory has the .so right next to
# __init__.py — which is the case for an editable install (cmake-build-extension
# drops the .so into the source tree) and for a wheel install (the .so lives in
# site-packages/warpkit/). Resolve the .so explicitly via importlib.util so the
# hook works regardless of install layout.
binaries = []
spec = importlib.util.find_spec("warpkit.warpkit_cpp")
if spec is not None and spec.origin:
    binaries.append((spec.origin, "warpkit"))

hiddenimports = ["warpkit.warpkit_cpp"]
