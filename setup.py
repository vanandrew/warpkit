import subprocess
from pathlib import Path
from setuptools import setup
from tempfile import TemporaryDirectory

# from cmake_setuptools_ext import CMakeExtension, CMakeBuild
from cmake_build_extension import CMakeExtension, BuildExtension


THISDIR = Path(__file__).parent

# get CMakeLists.txt
cmakelists = (Path(THISDIR) / "CMakeLists.txt").absolute().as_posix()
cmake_dir = (Path(THISDIR)).absolute().as_posix()

# get scripts path
scripts_path = THISDIR / "warpkit" / "scripts"

# ensure julia is installed
try:
    subprocess.run(["julia", "--version"], check=True)
except subprocess.CalledProcessError:
    raise ImportError("Julia is not installed. You need to install it and ensure it's on your PATH.")

# ensure ROMEO is installed
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
    raise OSError("ROMEO failed to install. Check your Julia installation.")

# open a temporary directory for install artifacts that we would like to remove
with TemporaryDirectory(prefix="build-tmp", dir=THISDIR / "warpkit") as tmpdir:
    setup(
        ext_modules=[
            CMakeExtension(
                name="warpkit.warpkit_cpp",
                source_dir=cmake_dir,
                cmake_configure_options=[
                    f"-DPYTHON_INSTALL_TMPDIR={Path(tmpdir).name}",
                ],
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
