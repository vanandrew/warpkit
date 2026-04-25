#ifndef ROMEO_VOLUME_VIEW_H
#define ROMEO_VOLUME_VIEW_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace romeo {

// Minimal non-owning column-major view over a 3D or 4D buffer.
// Column-major matches Julia's default layout and numpy's py::array::f_style.
//
// TODO(phase-2+): flesh out accessors. For now this is declaration-only so
// signatures can reference it without implementation.

template <typename T>
struct Volume3DView {
    T* data;
    std::array<std::size_t, 3> extent;   // (nx, ny, nz)
    std::array<std::ptrdiff_t, 3> stride;  // elements, not bytes
};

template <typename T>
struct Volume4DView {
    T* data;
    std::array<std::size_t, 4> extent;   // (nx, ny, nz, nt)
    std::array<std::ptrdiff_t, 4> stride;
};

}  // namespace romeo

#endif
