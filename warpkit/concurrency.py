from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Executor,
    Future,
    as_completed,
)
from threading import Lock
from typing import Callable, Iterator, Optional


class DummyExecutor(Executor):
    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn: Callable, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


def run_executor(
    ncpus: int,
    type: str,
    fn: Callable,
    iterator: Iterator,
    initializer: Optional[Callable] = None,
    post_fn: Optional[Callable] = None,
):
    """Runs executor with given number of cpus and type of executor.

    This function will create a concurrent.futures.Executor object and run the
    given function with the given arguments using the executor. If `ncpus` is
    set to 1, then the executor will be a DummyExecutor, which will run the
    function in the current thread. The `type` of executor can be set to either
    "thread" or "process", which will create a ThreadPoolExecutor or ProcessPoolExecutor,
    respectively. If `ncpus` is set to a number greater than 1, then the function
    will be run in parallel using the given number of threads or processes.

    Parameters
    ----------
    ncpus : int
        Number of cpus to use. If set to 1 then the function will be run in the
        current thread. If set to a number greater than 1, then the function
        will be run in parallel using the given number of threads or processes.
    type : str
        Type of executor to use. Can be either "thread" or "process".
    fn : Callable
        Function to call. The output of the iterator will be passed as unpacked
        arguments to this function.
    iterator : Iterator
        Iterator that yields the arguments to pass to the function.
    initializer : Callable, optional
        Function to call to initialize the executor.
    post_fn : Callable, optional
        Function to call after future has been unpacked.
    """
    # Create executor
    if ncpus == 1:
        if initializer is not None:
            initializer()
        executor = DummyExecutor()
    elif type == "thread":
        executor = ThreadPoolExecutor(ncpus, initializer=initializer)
    elif type == "process":
        executor = ProcessPoolExecutor(ncpus, initializer=initializer)
    else:
        raise ValueError("type must be either 'thread' or 'process'")

    # Create dict to store futures
    futures = dict()

    # Loop over iterator and submit jobs to executor
    for idx, args in enumerate(iterator):
        futures[executor.submit(fn, *args)] = idx

    # Loop over futures and unpack results
    for future in as_completed(futures):
        # get the index of the future
        idx = futures[future]
        # pass thre result of the future to the post_fn
        if post_fn is not None:
            post_fn(idx, future.result())

    # Shutdown executor
    executor.shutdown()
