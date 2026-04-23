import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from cmake_build_extension import BuildExtension, CMakeExtension  # type: ignore
from setuptools import setup

THISDIR = Path(__file__).parent
cmake_dir = (Path(THISDIR)).absolute().as_posix()
scripts_path = THISDIR / "warpkit" / "scripts"

IS_CIBUILDWHEEL = os.environ.get("CIBUILDWHEEL", "0") == "1"
IS_MACOS = os.environ.get("RUNNER_OS", "0") == "macOS"
python_version = sys.executable

with TemporaryDirectory(prefix="build-tmp-", dir="/tmp") as tmpdir:
    cmake_configure_options = [f"-DPYTHON_INSTALL_TMPDIR={tmpdir}", f"-DPython_EXECUTABLE={python_version}"]
    if IS_CIBUILDWHEEL and not IS_MACOS:
        cmake_configure_options.append("-DCIBUILDWHEEL=")
    if IS_MACOS:
        cmake_configure_options.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
    setup(
        ext_modules=[
            CMakeExtension(
                name="warpkit.warpkit_cpp",
                source_dir=cmake_dir,
                cmake_configure_options=cmake_configure_options,
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        entry_points={
            "console_scripts": [
                f"{f.stem}=warpkit.scripts.{f.stem}:main"
                for f in scripts_path.glob("*.py")
                if f.name not in "__init__.py"
            ]
        },
    )
