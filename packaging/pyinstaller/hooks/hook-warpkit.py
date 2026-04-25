import importlib.util
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, get_package_paths

# Resolve the C extension via Python's own importlib — this catches the .so
# regardless of whether warpkit is editable- or wheel-installed and whether
# the file matches PyInstaller's `lib*.so` default pattern (it does not, since
# the extension is `warpkit_cpp.cpython-*.so` — no `lib` prefix).
binaries = []
spec = importlib.util.find_spec("warpkit.warpkit_cpp")
if spec is not None and spec.origin:
    binaries.append((spec.origin, "warpkit"))

# Diagnostic prints so any future bundle-without-the-.so failure is debuggable
# from the build log.
print(
    f"[hook-warpkit] find_spec origin: {spec.origin if spec else None}", file=sys.stderr
)
print(
    f"[hook-warpkit] get_package_paths: {get_package_paths('warpkit')}", file=sys.stderr
)
print(
    f"[hook-warpkit] collect_dynamic_libs(*.so): "
    f"{collect_dynamic_libs('warpkit', search_patterns=['*.so', '*.dylib'])}",
    file=sys.stderr,
)
print(f"[hook-warpkit] binaries: {binaries}", file=sys.stderr)

hiddenimports = ["warpkit.warpkit_cpp"]
