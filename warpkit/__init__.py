from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("warpkit")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .warpkit_cpp import *  # noqa: F403  # pyright: ignore[reportMissingModuleSource]
