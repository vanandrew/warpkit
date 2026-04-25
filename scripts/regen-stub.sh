#!/usr/bin/env bash
# Regenerate warpkit/warpkit_cpp.pyi from the compiled extension and apply
# post-processing fixes. Run this after changing pybind11 bindings in
# src/warpkit.cpp.
#
# pybind11-stubgen 2.5.x emits `numpy.bool` for `py::array_t<bool>`, but
# `numpy.bool` was removed in NumPy 1.24. Rewrite to `numpy.bool_`.
set -e
cd "$(dirname "$0")/.."

uv run pybind11-stubgen warpkit.warpkit_cpp --output-dir .

# Avoid GNU/BSD sed differences: rewrite in place via Python.
uv run python -c '
import pathlib, re
p = pathlib.Path("warpkit/warpkit_cpp.pyi")
text = p.read_text()
text = re.sub(r"\bnumpy\.bool\b(?!_)", "numpy.bool_", text)
p.write_text(text)
'

uv run --with ruff ruff check --fix warpkit/warpkit_cpp.pyi || true
uv run --with ruff ruff format warpkit/warpkit_cpp.pyi
echo "Stub regenerated at warpkit/warpkit_cpp.pyi"
