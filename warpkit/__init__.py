try:
    from . import warpkit_cpp # type: ignore
except ImportError:
    import sys
    from pathlib import Path
    BUILD_DIR = str(Path(__file__).parent.parent / "build")
    sys.path.append(BUILD_DIR)
    import warpkit_cpp  # type: ignore
