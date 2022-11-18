import sys
import site
from setuptools import setup

# This line enables user based installation when using pip in editable mode with the latest
# pyproject.toml config.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# call setuptools setup
setup()
