from PyInstaller.utils.hooks import collect_dynamic_libs

# Default search_patterns is `lib*.so` (system-style libs) which misses
# `warpkit_cpp.cpython-*.so` since it has no `lib` prefix. The local repo
# build picks the .so up via PyInstaller's import-graph analysis, but a
# wheel-installed warpkit (what CI / users have) does not — so force the
# bundle here.
binaries = collect_dynamic_libs("warpkit", search_patterns=["*.so", "*.dylib", "*.dll"])
hiddenimports = ["warpkit.warpkit_cpp"]
