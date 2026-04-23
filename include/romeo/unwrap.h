#ifndef ROMEO_UNWRAP_H
#define ROMEO_UNWRAP_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "romeo/algorithm.h"
#include "romeo/utility.h"
#include "romeo/weights.h"

namespace romeo {

template <typename T>
struct Unwrap3DOptions {
    const T* mag = nullptr;        // (nx, ny, nz) or null
    const bool* mask = nullptr;    // (nx, ny, nz) or null
    bool correct_global = false;   // apply global 2π shift after MST
    int maxseeds = 1;
    T wrap_addition = T(0);
    // Second-echo context. When non-null, feeds both the
    // phase_gradient_coherence weight term and the multi-echo seed correction.
    // Only the 4D template-echo path sets these; standalone 3D unwrap leaves
    // them null.
    const T* phase2 = nullptr;
    T te1 = T(0);
    T te2 = T(0);
};

namespace detail {

// Julia Statistics.median: sort the sample; for even-length samples return the
// mean of the two middle elements. Implemented with std::nth_element so the
// full sort cost is amortized away on large volumes.
template <typename T>
inline T median_in_place(std::vector<T>& xs) {
    const std::size_t m = xs.size();
    const std::size_t mid = m / 2;
    std::nth_element(xs.begin(), xs.begin() + mid, xs.end());
    const T upper = xs[mid];
    if (m % 2 == 1) return upper;
    // For even sizes, the left partition's max is the other middle value.
    const T lower = *std::max_element(xs.begin(), xs.begin() + mid);
    return (upper + lower) / T(2);
}

// Subtract 2π * median(round(wrapped[mask ∧ finite] / 2π)) from every voxel.
// Ports the `correctglobal` branch of unwrap! in
// ROMEO.jl src/unwrapping.jl.
template <typename T>
inline void apply_correct_global(T* wrapped, std::size_t n, const bool* mask) {
    constexpr T two_pi = static_cast<T>(2.0L * 3.141592653589793238462643383279502884L);
    std::vector<T> rounded;
    rounded.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (mask != nullptr && !mask[i]) continue;
        if (!std::isfinite(wrapped[i])) continue;
        rounded.push_back(std::round(wrapped[i] / two_pi));
    }
    if (rounded.empty()) return;
    const T shift = two_pi * median_in_place(rounded);
    for (std::size_t i = 0; i < n; ++i) wrapped[i] -= shift;
}

}  // namespace detail

// 3D ROMEO unwrap — modifies `wrapped` in place.
//
// Ports `unwrap!(::AbstractArray{T,3}; ...)` from
// ROMEO.jl src/unwrapping.jl. Only the `:romeo` weight preset is
// supported; warpkit never requests any other. `merge_regions` and
// `correct_regions` are intentionally not implemented (warpkit always passes
// false).
template <typename T>
inline void unwrap_3d(T* wrapped, std::size_t nx, std::size_t ny, std::size_t nz, const Unwrap3DOptions<T>& opts) {
    const std::size_t n = nx * ny * nz;

    // Compute ROMEO edge weights for the :romeo preset.
    // When phase2 is provided we need TEs in a flat (te1, te2) pair to pass to
    // calculate_weights_romeo's phasegradientcoherence branch.
    const T tes_pair[2] = {opts.te1, opts.te2};
    const T* tes_ptr = (opts.phase2 != nullptr) ? tes_pair : nullptr;
    auto weights = calculate_weights_romeo<T>(wrapped, nx, ny, nz, opts.mag, opts.phase2, tes_ptr, opts.mask,
                                              romeo_flags_default());

    // Julia: `@assert sum(weights) != 0 "Unwrap-weights are all zero!"`
    // We turn this into a runtime_error so Python sees a useful message.
    bool any_edge = false;
    for (std::uint8_t w : weights) {
        if (w != 0) {
            any_edge = true;
            break;
        }
    }
    if (!any_edge) throw std::runtime_error("unwrap_3d: all edge weights are zero");

    // Grow the MST.
    std::vector<std::uint8_t> visited(n, 0);
    GrowRegionContext<T> ctx{wrapped,
                             nx,
                             ny,
                             nz,
                             static_cast<std::ptrdiff_t>(n),
                             {static_cast<std::ptrdiff_t>(1), static_cast<std::ptrdiff_t>(nx),
                              static_cast<std::ptrdiff_t>(nx * ny)},
                             weights.data(),
                             visited.data(),
                             opts.wrap_addition,
                             opts.phase2,
                             opts.te1,
                             opts.te2};
    grow_region_unwrap(ctx, opts.maxseeds);

    // Optional global 2π shift.
    if (opts.correct_global) {
        detail::apply_correct_global(wrapped, n, opts.mask);
    }
}

// 4D (multi-echo) ROMEO unwrap — modifies `wrapped` in place.
//
// Strategy (ports `unwrap!(::AbstractArray{T,4}; TEs, ...)` from
// ROMEO.jl src/unwrapping.jl, minus the `individual` and
// `temporal_uncertain_unwrapping` branches that warpkit never exercises):
//
//   1. Spatial-unwrap the template echo (default 0) using unwrap_3d, driven by
//      phase2 = wrapped[:,:,:, p2ref] and TEs = (TEs[template], TEs[p2ref])
//      where p2ref = 1 if template == 0 else template − 1.
//   2. For every other echo e, temporal-unwrap each voxel relative to a
//      reference echo iref (the neighboring echo closer to template):
//
//        ref = wrapped[:,:,:, iref] * (TEs[e] / TEs[iref])
//        wrapped[:,:,:, e] = unwrap_voxel(wrapped[:,:,:, e], ref)
//
//      Loop order mirrors Julia's `[(template-1):-1:1; (template+1):end]`:
//      walk outward from template, first descending, then ascending.
template <typename T>
inline void unwrap_4d(T* wrapped, std::size_t nx, std::size_t ny, std::size_t nz, std::size_t ne, const T* TEs,
                      const T* mag4d, const bool* mask, bool correct_global, int maxseeds,
                      std::size_t template_echo = 0) {
    if (ne < 2) throw std::invalid_argument("unwrap_4d: need at least 2 echoes");
    if (template_echo >= ne) throw std::invalid_argument("unwrap_4d: template_echo out of range");

    const std::size_t vol = nx * ny * nz;
    const std::size_t p2ref = (template_echo == 0) ? 1 : template_echo - 1;

    // --- 1. Spatial unwrap on the template echo ---
    T* template_view = wrapped + template_echo * vol;
    const T* phase2_view = wrapped + p2ref * vol;

    Unwrap3DOptions<T> opts;
    opts.mag = (mag4d != nullptr) ? (mag4d + template_echo * vol) : nullptr;
    opts.mask = mask;
    opts.correct_global = correct_global;
    opts.maxseeds = maxseeds;
    opts.wrap_addition = T(0);
    opts.phase2 = phase2_view;
    opts.te1 = TEs[template_echo];
    opts.te2 = TEs[p2ref];

    unwrap_3d<T>(template_view, nx, ny, nz, opts);

    // --- 2. Temporal unwrap for every non-template echo ---
    auto temporal = [&](std::size_t e) {
        const std::size_t iref = (e < template_echo) ? e + 1 : e - 1;
        const T ratio = TEs[e] / TEs[iref];
        T* e_view = wrapped + e * vol;
        const T* ref_view = wrapped + iref * vol;
        for (std::size_t i = 0; i < vol; ++i) {
            const T refval = ref_view[i] * ratio;
            e_view[i] = unwrap_voxel(e_view[i], refval);
        }
    };

    // Descending half: (template-1) .. 0.
    for (std::ptrdiff_t e = static_cast<std::ptrdiff_t>(template_echo) - 1; e >= 0; --e) {
        temporal(static_cast<std::size_t>(e));
    }
    // Ascending half: (template+1) .. ne-1.
    for (std::size_t e = template_echo + 1; e < ne; ++e) {
        temporal(e);
    }
}

}  // namespace romeo

#endif
