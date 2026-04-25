#ifndef ROMEO_VOXEL_QUALITY_H
#define ROMEO_VOXEL_QUALITY_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "romeo/weights.h"

namespace romeo {

// 4D voxel-quality map — port of `voxelquality()` + the 4D `calculateweights`
// dispatch in ROMEO.jl src/voxelquality.jl.
//
// Inputs:
//   - `phase4d` : column-major (nx, ny, nz, ne), ne ≥ 2
//   - `TEs`     : length-ne echo times
//   - `mag4d`   : optional (nx, ny, nz, ne) or null
//
// Strategy: take the first echo as `template` and the second as `p2ref` (the
// Julia defaults), call `calculate_weights_romeo_raw` with all four flags on
// (phasecoherence, phasegradientcoherence, phaselinearity, magcoherence;
// auto-disabled where inputs are missing), then aggregate: each voxel's qmap
// value is the sum of the three forward edges plus three backward edges
// (shifted reads from the neighbor voxel's forward-edge entry), divided by 6.
//
// Raw weights come back in [0, 1] with NaNs already mapped to 0 by the
// `raw` variant, so the qmap is finite by construction and lives in [0, 1].
template <typename T>
inline std::vector<T> voxel_quality(const T* phase4d, std::size_t nx, std::size_t ny, std::size_t nz, std::size_t ne,
                                    const T* TEs, const T* mag4d) {
    if (ne < 2) throw std::invalid_argument("voxel_quality: need at least 2 echoes");

    const std::size_t vol = nx * ny * nz;
    const std::size_t template_echo = 0;  // Julia: template=1 → 0-based 0.
    const std::size_t p2ref = 1;          // Julia default.

    const T* phase_t = phase4d + template_echo * vol;
    const T* phase2 = phase4d + p2ref * vol;
    const T tes_pair[2] = {TEs[template_echo], TEs[p2ref]};
    // If mag4d is 4D, slice the template echo — same view trick Julia uses.
    const T* mag_t = (mag4d != nullptr) ? (mag4d + template_echo * vol) : nullptr;

    // Raw [0,1] weights, shape (3, nx, ny, nz) column-major.
    const std::vector<T> w = calculate_weights_romeo_raw<T>(phase_t, nx, ny, nz, mag_t, phase2, tes_pair,
                                                             /*mask=*/nullptr, romeo_flags_default());

    std::vector<T> qmap(vol, T(0));

    // Single fused pass: each voxel sums its three forward edges plus the
    // forward-edge weight of its previous neighbor along each axis (= the
    // backward-edge contribution from Julia's three .+= expressions).
    // Associativity matches the original four-pass form: forward triplet
    // first, then back-x, then back-y, then back-z.
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t idx = i + nx * (j + ny * k);
                T q = w[0u + 3u * idx] + w[1u + 3u * idx] + w[2u + 3u * idx];
                if (i > 0) q += w[0u + 3u * (idx - 1)];
                if (j > 0) q += w[1u + 3u * (idx - nx)];
                if (k > 0) q += w[2u + 3u * (idx - nx * ny)];
                qmap[idx] = q / T(6);
            }
        }
    }
    return qmap;
}

}  // namespace romeo

#endif
