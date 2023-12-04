import logging
import os
from multiprocessing import Process
from . import JuliaContext as _JuliaContext  # type: ignore

# We start a JuliaContext so that we can check if ROMEO is installed.
# we have to make it in a separate process because Julia context can
# only be initialized once per process.
p = Process(target=_JuliaContext)
p.start()
p.join()


PID_MAP = {}


class JuliaContext:
    def __new__(cls):
        # we only want to start one JuliaContext per process
        pid = os.getpid()
        if pid not in PID_MAP:
            logging.info("Starting JuliaContext for process %d", pid)
            PID_MAP[pid] = _JuliaContext()
        return PID_MAP[pid]
