#include <pybind11/pybind11.h>
#include <romeo.h>
#include <warps.h>

// specify pybind namespace
namespace py = pybind11;

PYBIND11_MODULE(warpkit_cpp, m) {
    py::class_<JuliaContext<float>>(m, "JuliaContext")
        .def(py::init<>())
        .def("romeo_voxelquality", &JuliaContext<float>::romeo_voxelquality, "Wrapper for ROMEO voxelquality function",
             py::arg("phase"), py::arg("TEs"), py::arg("mag"), py::return_value_policy::move)
        .def("romeo_unwrap3D", &JuliaContext<float>::romeo_unwrap3D, "Wrapper for ROMEO unwrap_individual function",
             py::arg("phase"), py::arg("weights"), py::arg("mag"), py::arg("mask"), py::arg("correct_global") = true,
             py::arg("maxseeds") = 1, py::arg("merge_regions") = false, py::arg("correct_regions") = false,
             py::return_value_policy::move)
        .def("romeo_unwrap4D", &JuliaContext<float>::romeo_unwrap4D, "Wrapper for ROMEO unwrap_individual function",
             py::arg("phase"), py::arg("TEs"), py::arg("weights"), py::arg("mag"), py::arg("mask"),
             py::arg("correct_global") = true, py::arg("maxseeds") = 1, py::arg("merge_regions") = false,
             py::arg("correct_regions") = false, py::return_value_policy::move);

    m.def("invert_displacement_map", &invert_displacement_map<double>, "Invert a displacement map",
          py::arg("displacement_map"), py::arg("origin"), py::arg("direction"), py::arg("spacing"), py::arg("axis") = 1,
          py::arg("iterations") = 50, py::arg("verbose") = false, py::return_value_policy::move);

    m.def("invert_displacement_field", &invert_displacement_field<double>, "Invert a displacement field",
          py::arg("displacement_field"), py::arg("origin"), py::arg("direction"), py::arg("spacing"),
          py::arg("iterations") = 50, py::arg("verbose") = false, py::return_value_policy::move);

    m.def("compute_jacobian_determinant", &compute_jacobian_determinant<double>,
          "Compute the Jacobian determinant of a displacement field", py::arg("displacement_field"), py::arg("origin"),
          py::arg("direction"), py::arg("spacing"), py::return_value_policy::move);

    m.def("resample", &resample<double>, "Resample an image with transform", py::arg("input_image"),
          py::arg("input_origin"), py::arg("input_direction"), py::arg("input_spacing"), py::arg("output_shape"),
          py::arg("output_origin"), py::arg("output_direction"), py::arg("output_spacing"), py::arg("transform"),
          py::arg("transform_origin"), py::arg("transform_direction"), py::arg("transform_spacing"),
          py::return_value_policy::move);

    m.def("compute_hausdorff_distance", &compute_hausdorff_distance<double>, "Compute the Hausdorff Distance",
          py::arg("image1"), py::arg("image1_origin"), py::arg("image1_direction"), py::arg("image1_spacing"),
          py::arg("image2"), py::arg("image2_origin"), py::arg("image2_direction"), py::arg("image2_spacing"),
          py::return_value_policy::move);
}
