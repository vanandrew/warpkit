#include <pybind11/pybind11.h>

#include "romeo/romeo.h"
#include "warps.h"

namespace py = pybind11;

PYBIND11_MODULE(warpkit_cpp, m, py::mod_gil_not_used()) {
    m.def("calculate_weights", &romeo::calculate_weights<float>,
          "ROMEO edge-weight map (3, nx, ny, nz) uint8. Exposed for port validation; not used by warpkit.",
          py::arg("phase"),
          py::arg("mag") = py::array_t<float, py::array::f_style>(),
          py::arg("phase2") = py::array_t<float, py::array::f_style>(),
          py::arg("tes") = py::array_t<float, py::array::f_style>(),
          py::arg("mask") = py::array_t<bool, py::array::f_style>(),
          py::return_value_policy::move);

    m.def("romeo_voxelquality", &romeo::romeo_voxelquality<float>,
          "Compute a per-voxel quality map from multi-echo phase/magnitude", py::arg("phase"), py::arg("tes"),
          py::arg("mag"), py::return_value_policy::move);

    m.def("romeo_unwrap3d", &romeo::romeo_unwrap3D<float>, "3D ROMEO phase unwrap", py::arg("phase"),
          py::arg("weights"), py::arg("mag"), py::arg("mask"), py::arg("correct_global") = true,
          py::arg("maxseeds") = 1, py::arg("merge_regions") = false, py::arg("correct_regions") = false,
          py::return_value_policy::move);

    m.def("romeo_unwrap4d", &romeo::romeo_unwrap4D<float>, "4D (multi-echo) ROMEO phase unwrap", py::arg("phase"),
          py::arg("tes"), py::arg("weights"), py::arg("mag"), py::arg("mask"), py::arg("correct_global") = true,
          py::arg("maxseeds") = 1, py::arg("merge_regions") = false, py::arg("correct_regions") = false,
          py::return_value_policy::move);

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
