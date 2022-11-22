#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <romeo.h>
#include <utilities.h>

// specify pybind namespace
namespace py = pybind11;

// // unwrap phase data interface
// template <typename T, int N>
// py::array_t<T, N> unwrap(py::array_t<T, N> mag, py::array_t<T, N> phase) {
//     // get pointers to data
//     auto mag_ptr = const_cast<T*>(mag.data(0));
//     auto phase_ptr = const_cast<T*>(phase.data(0));

//     // get shape of each array
//     auto mag_shape = std::vector<ssize_t>(mag.shape(), mag.shape() + mag.ndim());
//     auto phase_shape = std::vector<ssize_t>(mag.shape(), mag.shape() + mag.ndim());
//     std::cout << "\n";
//     print(mag_shape);
//     print(phase_shape);
//     unwrap_individual();
//     return std::move(phase);
// }

PYBIND11_MODULE(mosaic_cpp, m) {
    py::class_<JuliaContext<double>>(m, "JuliaContext")
        .def(py::init<>())
        .def("romeo_unwrap_individual", &JuliaContext<double>::romeo_unwrap_individual,
             "Wrapper for ROMEO unwrap_individual function", py::arg("phase"), py::arg("TEs"), py::arg("weights"),
             py::arg("mag"), py::arg("mask"), py::arg("correctglobal") = false);
}
