#include <pybind11/pybind11.h>
#include <romeo.h>

// specify pybind namespace
namespace py = pybind11;

PYBIND11_MODULE(mosaic_cpp, m) {
    py::class_<JuliaContext<double>>(m, "JuliaContext")
        .def(py::init<>())
        .def("romeo_unwrap_individual", &JuliaContext<double>::romeo_unwrap_individual,
             "Wrapper for ROMEO unwrap_individual function", py::arg("phase"), py::arg("TEs"), py::arg("weights"),
             py::arg("mag"), py::arg("mask"), py::arg("correctglobal") = false);
}
