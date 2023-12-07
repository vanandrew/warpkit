import logging
import os
from . import JuliaContext as _JuliaContext  # type: ignore


PID_MAP = {}


class JuliaContext:
    def __new__(cls):
        # we only want to start one JuliaContext per process
        pid = os.getpid()
        if pid not in PID_MAP:
            logging.info("Starting JuliaContext for process %d", pid)
            PID_MAP[pid] = _JuliaContext()
        return PID_MAP[pid]
