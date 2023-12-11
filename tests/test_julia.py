from multiprocessing import Process, Manager
from warpkit.julia import JuliaContext


def test_julia_context():
    # these two should be the same object
    assert JuliaContext() is JuliaContext()
