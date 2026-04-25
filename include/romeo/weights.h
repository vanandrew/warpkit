#ifndef ROMEO_WEIGHTS_H
#define ROMEO_WEIGHTS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "romeo/utility.h"

namespace romeo {

// Per-edge weight flags matching flags 1..4 of
// ROMEO.jl src/weights.jl. The `:romeo` preset (the only one warpkit
// uses) turns on all four; flags that don't have their inputs available are
// forced off inside `calculate_weights_romeo` below (see `updateflags!` in the
// Julia source).
//
// Flags 5..6 (magweight / magweight2) are intentionally not ported.
struct RomeoWeightFlags {
    bool phase_coherence;
    bool phase_gradient_coherence;
    bool phase_linearity;
    bool mag_coherence;
};

inline RomeoWeightFlags romeo_flags_default() {
    return RomeoWeightFlags{true, true, true, true};
}

// Rescale a raw weight w ∈ [0, 1] to a bin in [1, NBINS]; any other value
// (including NaN) maps to 0 which means "edge invalid, not enqueued".
// Matches `rescale(w)` in ROMEO.jl src/weights.jl.
template <typename T>
inline int rescale_weight(T w) {
    if (!(w >= T(0)) || !(w <= T(1))) return 0;
    const T raw = (T(1) - w) * T(NBINS - 1);
    const int bin = static_cast<int>(std::lround(raw));
    return bin < 1 ? 1 : bin;
}

// Fundamental coherence/linearity terms. Direct C++ translations of the
// corresponding one-liners in ROMEO.jl src/weights.jl.
namespace detail {

template <typename T>
inline T phase_coherence(T a, T b) {
    constexpr T pi = static_cast<T>(3.141592653589793238462643383279502884L);
    return T(1) - std::abs(gamma_fold(a - b) / pi);
}

template <typename T>
inline T phase_gradient_coherence(T a, T b, T a2, T b2, T te1, T te2) {
    T g1 = gamma_fold(a - b);
    T g2 = gamma_fold(a2 - b2);
    T ratio = te1 / te2;
    T val = T(1) - std::abs(g1 - g2 * ratio);
    return std::max<T>(T(0), val);
}

template <typename T>
inline T mag_coherence(T small, T big) {
    T r = small / big;
    return r * r;
}

// Three-point phase linearity: Julia's `phaselinearity(P, i, j, k)`.
// The NaN guard matches the post-v1.0.0 ROMEO.jl master — any NaN operand
// collapses the triplet to 0.5 rather than propagating. Upstream's own
// specialcases.jl gates the NaN golden on Julia 1.8+ for exactly this reason.
//
// Subtle detail: Julia's `max(0, NaN)` returns NaN, so its `isnan(pl)` check
// catches it after the max. C++'s `std::max(0, NaN)` returns 0 (NaN compares
// as !(<), so the "else" branch wins), which would silently hide the NaN.
// We therefore check for NaN on the pre-max value.
template <typename T>
inline T phase_linearity_triplet(T pi_v, T pj_v, T pk_v) {
    T s = pi_v - T(2) * pj_v + pk_v;
    T r = rem2pi_nearest(s);
    T val = T(1) - std::abs(r / T(2));
    if (std::isnan(val)) return T(0.5);
    return std::max<T>(T(0), val);
}

// Two-point dispatch: Julia's `phaselinearity(P, i, j)`. If the neighbors on
// both sides of the edge exist, return the product of the two triplets;
// otherwise return the border-penalty value 0.9.
//
// `stride` is the signed step in 1D linear index from `i` to `j`. Because the
// weight loop visits each dim with a fixed positive stride, `stride` is always
// +1, +nx, or +nx*ny. `n` is the total buffer length so we can range-check.
template <typename T>
inline T phase_linearity(const T* phase, std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t stride,
                         std::ptrdiff_t n) {
    std::ptrdiff_t h = i - stride;
    std::ptrdiff_t k = j + stride;
    if (h < 0 || k >= n) return static_cast<T>(0.9);
    return phase_linearity_triplet(phase[h], phase[i], phase[j]) *
           phase_linearity_triplet(phase[i], phase[j], phase[k]);
}

}  // namespace detail

// Generic weight-loop shared by the uint8-bin (unwrap) and raw-float
// (voxelquality) paths. `Rescale` is a callable T→OutT that mirrors
// ROMEO.jl's `rescale` kwarg. `zero_value` is the sentinel written for
// out-of-bounds or masked edges (0 for both output types).
//
// Mirrors `calculateweights_romeo(wrapped, flags)` + the edge-zeroing tail of
// `calculateweights` in ROMEO.jl src/weights.jl.
template <typename OutT, typename T, typename Rescale>
inline std::vector<OutT> calculate_weights_romeo_impl(const T* phase, std::size_t nx, std::size_t ny, std::size_t nz,
                                                     const T* mag, const T* phase2, const T* TEs, const bool* mask,
                                                     RomeoWeightFlags flags, Rescale rescale_fn, OutT zero_value) {
    // Apply `updateflags!` — turn off any term whose inputs are missing.
    if (mag == nullptr) flags.mag_coherence = false;
    if (phase2 == nullptr || TEs == nullptr) flags.phase_gradient_coherence = false;

    const std::size_t n = nx * ny * nz;
    const std::ptrdiff_t signed_n = static_cast<std::ptrdiff_t>(n);
    const std::ptrdiff_t sx = 1;
    const std::ptrdiff_t sy = static_cast<std::ptrdiff_t>(nx);
    const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(nx * ny);

    std::vector<OutT> weights(3 * n, zero_value);

    // Mirrors ROMEO's `mag .* mask` pre-pass without materializing a copy: any
    // masked-out voxel reads as 0 magnitude.
    auto mag_at = [&](std::ptrdiff_t i) -> T {
        if (mask != nullptr && !mask[i]) return T(0);
        return mag[i];
    };

    auto edge_weight = [&](std::ptrdiff_t idx, std::ptrdiff_t jdx, std::ptrdiff_t stride) -> T {
        T w = T(1);
        if (flags.phase_coherence) {
            w *= T(0.1) + T(0.9) * detail::phase_coherence(phase[idx], phase[jdx]);
        }
        if (flags.phase_gradient_coherence) {
            w *= T(0.1) + T(0.9) * detail::phase_gradient_coherence(phase[idx], phase[jdx], phase2[idx], phase2[jdx],
                                                                    TEs[0], TEs[1]);
        }
        if (flags.phase_linearity) {
            w *= T(0.1) + T(0.9) * detail::phase_linearity(phase, idx, jdx, stride, signed_n);
        }
        if (mag != nullptr) {
            T mi = mag_at(idx);
            T mj = mag_at(jdx);
            T small = std::min(mi, mj);
            T big = std::max(mi, mj);
            if (flags.mag_coherence) {
                w *= T(0.1) + T(0.9) * detail::mag_coherence(small, big);
            }
            // flags 5-6 (magweight, magweight2) not ported — warpkit uses :romeo, not :romeo6.
        }
        return w;
    };

    // One pass over voxels in (k, j, i) order. Boundary edges are skipped by
    // construction via the `+1 < n*` guards, so the trailing zero-fill pass
    // from the Julia source is unnecessary — the value-init of `weights`
    // already leaves those slots at `zero_value`.
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t idx = i + nx * (j + ny * k);
                if (mask != nullptr && !mask[idx]) continue;
                const std::ptrdiff_t sidx = static_cast<std::ptrdiff_t>(idx);
                if (i + 1 < nx) {
                    weights[0u + 3u * idx] = rescale_fn(edge_weight(sidx, sidx + sx, sx));
                }
                if (j + 1 < ny) {
                    weights[1u + 3u * idx] = rescale_fn(edge_weight(sidx, sidx + sy, sy));
                }
                if (k + 1 < nz) {
                    weights[2u + 3u * idx] = rescale_fn(edge_weight(sidx, sidx + sz, sz));
                }
            }
        }
    }

    return weights;
}

// Integer-bin weights (unwrap path). Returns (3, nx, ny, nz) uint8; 0 = edge
// invalid / do not enqueue; otherwise weight ∈ [1, NBINS].
template <typename T>
inline std::vector<std::uint8_t> calculate_weights_romeo(const T* phase, std::size_t nx, std::size_t ny,
                                                          std::size_t nz, const T* mag, const T* phase2,
                                                          const T* TEs, const bool* mask, RomeoWeightFlags flags) {
    auto to_bin = [](T w) -> std::uint8_t { return static_cast<std::uint8_t>(rescale_weight(w)); };
    return calculate_weights_romeo_impl<std::uint8_t, T>(phase, nx, ny, nz, mag, phase2, TEs, mask, flags, to_bin,
                                                         std::uint8_t{0});
}

// Raw float weights (voxelquality path). Returns (3, nx, ny, nz) T in [0, 1].
// Out-of-range / NaN raw weights collapse to 0 — this matches how Julia's
// `rescale` maps invalid values and keeps the qmap finite without a separate
// NaN-replacement pass.
template <typename T>
inline std::vector<T> calculate_weights_romeo_raw(const T* phase, std::size_t nx, std::size_t ny, std::size_t nz,
                                                    const T* mag, const T* phase2, const T* TEs, const bool* mask,
                                                    RomeoWeightFlags flags) {
    auto clamp_identity = [](T w) -> T {
        if (!(w >= T(0)) || !(w <= T(1))) return T(0);
        return w;
    };
    return calculate_weights_romeo_impl<T, T>(phase, nx, ny, nz, mag, phase2, TEs, mask, flags, clamp_identity, T(0));
}

}  // namespace romeo

#endif
