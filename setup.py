from pathlib import Path
from setuptools import setup
from cmake_setuptools_ext import CMakeExtension, CMakeBuild


THISDIR = Path(__file__).parent

# get CMakeLists.txt
cmakelists = (Path(THISDIR) / "CMakeLists.txt").absolute().as_posix()

# get scripts path
scripts_path = THISDIR / "mosaic" / "scripts"

setup(
    ext_modules=[CMakeExtension("mosaic.moasic_cpp", cmakelists)],
    cmdclass={"build_ext": CMakeBuild},
    entry_points={
        "console_scripts": [
            f"{f.stem}=mosaic.scripts.{f.stem}:main"
            for f in scripts_path.glob("*.py")
            if f.name not in "__init__.py"
        ]
    },
)
