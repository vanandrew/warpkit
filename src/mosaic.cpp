#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

extern "C" {
    #include <romeo.h>
}

// specify pybind namespace
namespace py = pybind11;

int unwrap_individual() {
    jl_unwrap_individual();
    return 0;
}

PYBIND11_MODULE(mosaic_cpp, m) {
    m.def("unwrap_individual", &unwrap_individual, py::return_value_policy::move);
}
