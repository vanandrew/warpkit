import numpy as np
import pytest
from warpkit.concurrency import DummyExecutor, run_executor


def _add(x, y):
    return x + y


def _raise():
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# DummyExecutor unit tests
# ---------------------------------------------------------------------------


def test_dummy_executor_submit_returns_future_with_result():
    executor = DummyExecutor()
    fut = executor.submit(_add, 2, 3)
    assert fut.result() == 5


def test_dummy_executor_captures_exceptions_on_future():
    executor = DummyExecutor()
    fut = executor.submit(_raise)
    with pytest.raises(RuntimeError, match="boom"):
        fut.result()


def test_dummy_executor_rejects_submit_after_shutdown():
    executor = DummyExecutor()
    executor.shutdown()
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        executor.submit(_add, 1, 2)


# ---------------------------------------------------------------------------
# run_executor across all three executor backends
# ---------------------------------------------------------------------------


def _gather_results(ncpus, executor_type):
    out = np.zeros(10)

    def gen(n):
        for i in range(n):
            yield (i, 1)

    def post(idx, result):
        out[idx] = result

    run_executor(ncpus, executor_type, _add, gen(10), post_fn=post)
    return out


@pytest.mark.parametrize(
    "ncpus,executor_type",
    [(1, "thread"), (2, "thread"), (2, "process")],
)
def test_run_executor_backends(ncpus, executor_type):
    out = _gather_results(ncpus, executor_type)
    assert np.array_equal(out, np.arange(10) + 1)


def test_run_executor_rejects_unknown_type():
    with pytest.raises(ValueError, match="must be either"):
        run_executor(2, "magic", _add, iter([(1, 1)]))


def test_run_executor_calls_initializer_in_single_cpu_path():
    """ncpus=1 takes the DummyExecutor branch, which manually invokes the
    initializer once before submitting any work."""
    calls = []

    def init():
        calls.append("init")

    run_executor(1, "thread", _add, iter([(1, 1), (2, 2)]), initializer=init)
    assert calls == ["init"]


def test_run_executor_no_post_fn_smoke():
    """post_fn=None is allowed: results are simply discarded."""
    run_executor(1, "thread", _add, iter([(1, 2)]))
