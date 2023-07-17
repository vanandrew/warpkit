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
