import logging
import os
from . import JuliaContext as _JuliaContext  # type: ignore

# There is a known bug where Julia will crash if you start a JuliaContext in
# the current processm, then fork a child process. I currently don't have a way
# to handle this atm other than to tell the user to not do that...
# In practice, this should be a very rare occurence, (why would you run this with
# a single process, then again with multiple processes in the same run?)


PID_MAP = {}


class JuliaContext:
    def __new__(cls):
        # we only want to start one JuliaContext per process
        pid = os.getpid()
        if pid not in PID_MAP:
            logging.info("Starting JuliaContext for process %d", pid)
            PID_MAP[pid] = _JuliaContext()
        return PID_MAP[pid]
