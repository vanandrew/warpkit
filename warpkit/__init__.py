try:
    from .warpkit_cpp import *  # type: ignore
except ImportError:
    import sys
    from pathlib import Path
    BUILD_DIR = str(Path(__file__).parent.parent / "build")
    sys.path.append(BUILD_DIR)
    from warpkit_cpp import *  # type: ignore
