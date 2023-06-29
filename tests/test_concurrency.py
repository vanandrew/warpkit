import numpy as np
import pytest
from warpkit.concurrency import run_executor


# define a test function
def dummy_fn(x, y):
    return x + y


def test_run_executor():
    # create an array to store the result
    result_array = np.zeros(10)

    # define a test iterator
    def test_iterator(num):
        for i in range(num):
            yield (i, 1)

    # define a post_fn function for storing result
    def test_post_fn(idx, result):
        result_array[idx] = result

    run_executor(1, "thread", dummy_fn, test_iterator(10), post_fn=test_post_fn)
    assert np.all(result_array == np.arange(10) + 1)
    run_executor(2, "thread", dummy_fn, test_iterator(10), post_fn=test_post_fn)
    assert np.all(result_array == np.arange(10) + 1)
    run_executor(2, "process", dummy_fn, test_iterator(10), post_fn=test_post_fn)
    assert np.all(result_array == np.arange(10) + 1)
