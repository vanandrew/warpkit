from multiprocessing import Process, Manager
from warpkit.julia import JuliaContext


def test_julia_context():
    # define state variable for Context state
    state = Manager().Value(bool, False)

    # start contexts in new processes so we don't have to worry about
    # forking a JuliaContext later on
    def start_contexts(state):
        Julia = JuliaContext()
        Julia2 = JuliaContext()
        # these two should be the same object
        state.value = Julia is Julia2

    # spawn process
    p = Process(target=start_contexts, args=(state,))
    p.start()
    p.join()
    # these two should be the same object
    assert state.value
