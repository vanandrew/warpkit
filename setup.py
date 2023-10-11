import subprocess
from pathlib import Path
from setuptools import setup
from cmake_setuptools_ext import CMakeExtension, CMakeBuild


THISDIR = Path(__file__).parent

# get CMakeLists.txt
cmakelists = (Path(THISDIR) / "CMakeLists.txt").absolute().as_posix()

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

setup(
    ext_modules=[CMakeExtension("warpkit.warpkit_cpp", cmakelists)],
    cmdclass={"build_ext": CMakeBuild},
    entry_points={
        "console_scripts": [
            f"{f.stem}=warpkit.scripts.{f.stem}:main" for f in scripts_path.glob("*.py") if f.name not in "__init__.py"
        ]
    },
)
