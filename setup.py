import subprocess
import os
import sys
from pathlib import Path
from setuptools import setup
from tempfile import TemporaryDirectory
from cmake_build_extension import CMakeExtension, BuildExtension

# get this directory
THISDIR = Path(__file__).parent

# get CMakeLists.txt
cmakelists = (Path(THISDIR) / "CMakeLists.txt").absolute().as_posix()
cmake_dir = (Path(THISDIR)).absolute().as_posix()

# get scripts path
scripts_path = THISDIR / "warpkit" / "scripts"

# is cibuildwheel environment?
IS_CIBUILDWHEEL = os.environ.get("CIBUILDWHEEL", "0") == "1"

# is this macOS?
IS_MACOS = os.environ.get("RUNNER_OS", "0") == "macOS"

# set python version to give hint for cmake
python_version = sys.executable

# ensure julia is installed
try:
    subprocess.run(["julia", "--startup-file=no", "--version"], check=True)
except subprocess.CalledProcessError:
    raise ImportError("Julia is not installed. You need to install it and ensure it's on your PATH.")

# ensure julia packages are installed
# only if not cibuildwheel (enabling this will cause cibuildwheel to fail)
if not IS_CIBUILDWHEEL:
    try:
        subprocess.run(
            [
                "julia",
                "-e",
                (
                    'using Pkg; !in("ROMEO",'
                    "[dep.name for (uuid, dep) in Pkg.dependencies()])"
                    ' ? Pkg.add(Pkg.PackageSpec(;name="ROMEO", version="1.0.0")) : nothing'
                ),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        raise ImportError("ROMEO is not installed. You need to install it and ensure it's on your PATH.")

# open a temporary directory for install artifacts that we would like to remove
with TemporaryDirectory(prefix="build-tmp-", dir="/tmp") as tmpdir:
    cmake_configure_options = [f"-DPYTHON_INSTALL_TMPDIR={tmpdir}", f"-DPython_EXECUTABLE={python_version}"]
    if IS_CIBUILDWHEEL and not IS_MACOS:
        cmake_configure_options.append(f"-DCIBUILDWHEEL=")
    if IS_MACOS:
        cmake_configure_options.append(f"-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
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
