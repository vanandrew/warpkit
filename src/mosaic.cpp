#include <pybind11/pybind11.h>
#include <romeo.h>
#include <warps.h>

// specify pybind namespace
namespace py = pybind11;

PYBIND11_MODULE(warpkit_cpp, m) {
    py::class_<JuliaContext<double>>(m, "JuliaContext")
        .def(py::init<>())
        .def("romeo_unwrap_individual", &JuliaContext<double>::romeo_unwrap_individual,
             "Wrapper for ROMEO unwrap_individual function", py::arg("phase"), py::arg("TEs"), py::arg("weights"),
             py::arg("mag"), py::arg("mask"), py::arg("correct_global") = true, py::return_value_policy::move);

    m.def("invert_displacement_map", &invert_displacement_map<double>, "Invert a displacement map",
          py::arg("displacement_map"), py::arg("origin"), py::arg("direction"), py::arg("spacing"), py::arg("axis") = 1,
          py::arg("iterations") = 50, py::arg("verbose") = false, py::return_value_policy::move);

    m.def("invert_displacement_field", &invert_displacement_field<double>, "Invert a displacement field",
          py::arg("displacement_field"), py::arg("origin"), py::arg("direction"), py::arg("spacing"),
          py::arg("iterations") = 50, py::arg("verbose") = false, py::return_value_policy::move);
}
