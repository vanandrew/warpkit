from warpkit.julia import JuliaContext


def test_julia_context():
    Julia = JuliaContext()
    Julia2 = JuliaContext()
    # these two should be the same object
    assert Julia is Julia2
