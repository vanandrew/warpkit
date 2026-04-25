#ifndef ROMEO_ROMEO_H
#define ROMEO_ROMEO_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "romeo/unwrap.h"
#include "romeo/voxel_quality.h"
#include "romeo/volume_view.h"
#include "romeo/weights.h"
#include "utilities.h"

namespace py = pybind11;

namespace romeo {

// Python-facing entry points over the pure-C++ ROMEO implementation.
//
// Function names retain the `romeo_*` prefix from the original Julia-backed
// API so call sites in warpkit/unwrap.py read naturally. They are stateless
// free functions; pybind11 binds them as module-level functions in
// `warpkit_cpp` (no class wrapper).

// Port of ROMEO.jl `calculateweights(phase; weights=:romeo, ...)` — the
// only weight preset we support. Exposed primarily so Python tests can
// validate the internal machinery against literal goldens from
// ROMEO.jl test/specialcases.jl. Not used by warpkit itself.
//
// `phase` is a column-major (nx, ny, nz) array. `mag`, `phase2`, `mask` may
// be 0-sized / empty arrays to indicate "not provided"; `TEs` is required
// only when `phase2` is provided (length 2: [te_phase, te_phase2]).
// Returns a (3, nx, ny, nz) uint8 array.
template <typename T>
py::array_t<std::uint8_t, py::array::f_style> calculate_weights(py::array_t<T, py::array::f_style> phase,
                                                                py::array_t<T, py::array::f_style> mag,
                                                                py::array_t<T, py::array::f_style> phase2,
                                                                py::array_t<T, py::array::f_style> TEs,
                                                                py::array_t<bool, py::array::f_style> mask) {
    if (phase.ndim() != 3) throw std::invalid_argument("calculate_weights: phase must be 3D");
    const auto nx = static_cast<std::size_t>(phase.shape(0));
    const auto ny = static_cast<std::size_t>(phase.shape(1));
    const auto nz = static_cast<std::size_t>(phase.shape(2));
    const auto n = nx * ny * nz;

    auto shape_matches = [&](const auto& arr) {
        return arr.ndim() == 3 && static_cast<std::size_t>(arr.shape(0)) == nx &&
               static_cast<std::size_t>(arr.shape(1)) == ny && static_cast<std::size_t>(arr.shape(2)) == nz;
    };

    const T* mag_ptr = mag.size() > 0 ? mag.data() : nullptr;
    if (mag_ptr && !shape_matches(mag)) throw std::invalid_argument("calculate_weights: mag shape mismatch");

    const T* phase2_ptr = phase2.size() > 0 ? phase2.data() : nullptr;
    if (phase2_ptr && !shape_matches(phase2)) throw std::invalid_argument("calculate_weights: phase2 shape mismatch");

    const T* tes_ptr = TEs.size() > 0 ? TEs.data() : nullptr;
    if (phase2_ptr && (!tes_ptr || TEs.size() < 2))
        throw std::invalid_argument("calculate_weights: TEs (length 2) required when phase2 is provided");

    const bool* mask_ptr = mask.size() > 0 ? mask.data() : nullptr;
    if (mask_ptr && !shape_matches(mask)) throw std::invalid_argument("calculate_weights: mask shape mismatch");

    auto w = calculate_weights_romeo<T>(phase.data(), nx, ny, nz, mag_ptr, phase2_ptr, tes_ptr, mask_ptr,
                                        romeo_flags_default());

    // Build a column-major (3, nx, ny, nz) numpy array and copy into it.
    py::array_t<std::uint8_t, py::array::f_style> out({static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(nx),
                                                       static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nz)});
    std::uint8_t* out_ptr = out.mutable_data();
    const std::size_t total = 3u * n;
    for (std::size_t i = 0; i < total; ++i) out_ptr[i] = w[i];
    return out;
}

template <typename T>
py::array_t<T, py::array::f_style> romeo_voxelquality(py::array_t<T, py::array::f_style> phase,
                                                      py::array_t<T, py::array::f_style> TEs,
                                                      py::array_t<T, py::array::f_style> mag) {
    if (phase.ndim() != 4) throw std::invalid_argument("romeo_voxelquality: phase must be 4D");
    const auto nx = static_cast<std::size_t>(phase.shape(0));
    const auto ny = static_cast<std::size_t>(phase.shape(1));
    const auto nz = static_cast<std::size_t>(phase.shape(2));
    const auto ne = static_cast<std::size_t>(phase.shape(3));

    if (TEs.ndim() != 1 || static_cast<std::size_t>(TEs.shape(0)) != ne)
        throw std::invalid_argument("romeo_voxelquality: TEs must be 1D with length equal to number of echoes");

    const T* mag_ptr = nullptr;
    if (mag.size() > 0) {
        if (mag.ndim() != 4 || static_cast<std::size_t>(mag.shape(0)) != nx ||
            static_cast<std::size_t>(mag.shape(1)) != ny ||
            static_cast<std::size_t>(mag.shape(2)) != nz || static_cast<std::size_t>(mag.shape(3)) != ne)
            throw std::invalid_argument("romeo_voxelquality: mag shape must match phase");
        mag_ptr = mag.data();
    }

    auto qmap_vec = voxel_quality<T>(phase.data(), nx, ny, nz, ne, TEs.data(), mag_ptr);

    py::array_t<T, py::array::f_style> out({static_cast<py::ssize_t>(nx), static_cast<py::ssize_t>(ny),
                                            static_cast<py::ssize_t>(nz)});
    T* out_ptr = out.mutable_data();
    const std::size_t vol = nx * ny * nz;
    for (std::size_t i = 0; i < vol; ++i) out_ptr[i] = qmap_vec[i];
    return out;
}

template <typename T>
py::array_t<T, py::array::f_style> romeo_unwrap3D(py::array_t<T, py::array::f_style> phase,
                                                  std::string weights,
                                                  py::array_t<T, py::array::f_style> mag,
                                                  py::array_t<bool, py::array::f_style> mask,
                                                  bool correctglobal = false,
                                                  int maxseeds = 1,
                                                  bool merge_regions = false,
                                                  bool correct_regions = false) {
    if (weights != "romeo")
        throw std::invalid_argument("romeo_unwrap3D: only the \"romeo\" weight preset is supported.");
    if (merge_regions || correct_regions)
        throw std::invalid_argument("romeo_unwrap3D: merge_regions / correct_regions are not implemented.");
    if (phase.ndim() != 3) throw std::invalid_argument("romeo_unwrap3D: phase must be 3D");

    const auto nx = static_cast<std::size_t>(phase.shape(0));
    const auto ny = static_cast<std::size_t>(phase.shape(1));
    const auto nz = static_cast<std::size_t>(phase.shape(2));
    const auto n = nx * ny * nz;

    auto shape_matches_3d = [&](const auto& arr) {
        return arr.ndim() == 3 && static_cast<std::size_t>(arr.shape(0)) == nx &&
               static_cast<std::size_t>(arr.shape(1)) == ny && static_cast<std::size_t>(arr.shape(2)) == nz;
    };

    const T* mag_ptr = mag.size() > 0 ? mag.data() : nullptr;
    if (mag_ptr && !shape_matches_3d(mag)) throw std::invalid_argument("romeo_unwrap3D: mag shape mismatch");
    const bool* mask_ptr = mask.size() > 0 ? mask.data() : nullptr;
    if (mask_ptr && !shape_matches_3d(mask)) throw std::invalid_argument("romeo_unwrap3D: mask shape mismatch");

    // Allocate output (column-major copy of `phase`), then unwrap in place.
    py::array_t<T, py::array::f_style> out({static_cast<py::ssize_t>(nx), static_cast<py::ssize_t>(ny),
                                            static_cast<py::ssize_t>(nz)});
    T* out_ptr = out.mutable_data();
    const T* phase_ptr = phase.data();
    for (std::size_t i = 0; i < n; ++i) out_ptr[i] = phase_ptr[i];

    Unwrap3DOptions<T> opts;
    opts.mag = mag_ptr;
    opts.mask = mask_ptr;
    opts.correct_global = correctglobal;
    opts.maxseeds = maxseeds;
    opts.wrap_addition = T(0);
    // Standalone 3D unwrap: no phase2/TEs.
    unwrap_3d<T>(out_ptr, nx, ny, nz, opts);
    return out;
}

template <typename T>
py::array_t<T, py::array::f_style> romeo_unwrap4D(py::array_t<T, py::array::f_style> phase,
                                                  py::array_t<T, py::array::f_style> TEs,
                                                  std::string weights,
                                                  py::array_t<T, py::array::f_style> mag,
                                                  py::array_t<bool, py::array::f_style> mask,
                                                  bool correctglobal = false,
                                                  int maxseeds = 1,
                                                  bool merge_regions = false,
                                                  bool correct_regions = false) {
    if (weights != "romeo")
        throw std::invalid_argument("romeo_unwrap4D: only the \"romeo\" weight preset is supported.");
    if (merge_regions || correct_regions)
        throw std::invalid_argument("romeo_unwrap4D: merge_regions / correct_regions are not implemented.");
    if (phase.ndim() != 4) throw std::invalid_argument("romeo_unwrap4D: phase must be 4D");

    const auto nx = static_cast<std::size_t>(phase.shape(0));
    const auto ny = static_cast<std::size_t>(phase.shape(1));
    const auto nz = static_cast<std::size_t>(phase.shape(2));
    const auto ne = static_cast<std::size_t>(phase.shape(3));
    const auto n_total = nx * ny * nz * ne;

    if (TEs.ndim() != 1 || static_cast<std::size_t>(TEs.shape(0)) != ne)
        throw std::invalid_argument("romeo_unwrap4D: TEs must be 1D with length equal to number of echoes");

    // mag: expected 4D (nx, ny, nz, ne). Empty means "no magnitudes".
    const T* mag_ptr = nullptr;
    if (mag.size() > 0) {
        if (mag.ndim() != 4 || static_cast<std::size_t>(mag.shape(0)) != nx ||
            static_cast<std::size_t>(mag.shape(1)) != ny ||
            static_cast<std::size_t>(mag.shape(2)) != nz || static_cast<std::size_t>(mag.shape(3)) != ne)
            throw std::invalid_argument("romeo_unwrap4D: mag shape must match phase");
        mag_ptr = mag.data();
    }

    // mask: 3D (nx, ny, nz). Empty means "no mask".
    const bool* mask_ptr = nullptr;
    if (mask.size() > 0) {
        if (mask.ndim() != 3 || static_cast<std::size_t>(mask.shape(0)) != nx ||
            static_cast<std::size_t>(mask.shape(1)) != ny || static_cast<std::size_t>(mask.shape(2)) != nz)
            throw std::invalid_argument("romeo_unwrap4D: mask must be 3D matching phase spatial dims");
        mask_ptr = mask.data();
    }

    py::array_t<T, py::array::f_style> out({static_cast<py::ssize_t>(nx), static_cast<py::ssize_t>(ny),
                                            static_cast<py::ssize_t>(nz), static_cast<py::ssize_t>(ne)});
    T* out_ptr = out.mutable_data();
    const T* phase_ptr = phase.data();
    for (std::size_t i = 0; i < n_total; ++i) out_ptr[i] = phase_ptr[i];

    // Julia's default template echo is 1 (first echo). In 0-based: 0.
    unwrap_4d<T>(out_ptr, nx, ny, nz, ne, TEs.data(), mag_ptr, mask_ptr, correctglobal, maxseeds,
                 /*template_echo=*/0);
    return out;
}

}  // namespace romeo

#endif
