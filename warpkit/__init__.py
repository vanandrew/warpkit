import os
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

if os.environ.get("WARPKIT_DEV", False) == "1":
    from warnings import warn

    warn("WARPKIT_DEV is set, trying to load from repo root build directory instead...")
    import sys
    from pathlib import Path

    BUILD_DIR = str(Path(__file__).parent.parent / "build")
    sys.path.append(BUILD_DIR)
    from warpkit_cpp import *  # type: ignore

    warn("Successfully loaded warpkit_cpp from repo root build directory!")
else:
    from .warpkit_cpp import *  # type: ignore
