# try to load julia ROMEO module
import subprocess
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

# import warpkit_cpp
try:
    from .warpkit_cpp import *  # type: ignore
except ImportError:
    print("Importing warpkit_cpp failed, trying to load from repo root build directory instead...")
    import sys
    from pathlib import Path

    BUILD_DIR = str(Path(__file__).parent.parent / "build")
    sys.path.append(BUILD_DIR)
    from warpkit_cpp import *  # type: ignore

    print("Successfully loaded warpkit_cpp from repo root build directory!")
