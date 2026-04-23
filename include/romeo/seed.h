#ifndef ROMEO_SEED_H
#define ROMEO_SEED_H

#include <cmath>
#include <cstddef>
#include <limits>

#include "romeo/utility.h"

namespace romeo {

// Single-echo seed correction — snap the seed voxel's phase into (-π, π].
// Ports the `else` branch of `seedcorrection!` in third_party/ROMEO/src/seed.jl.
template <typename T>
inline void seed_correction_single(T* wrapped, std::ptrdiff_t vox) {
    wrapped[vox] = rem2pi_nearest(wrapped[vox]);
}

// Multi-echo seed correction — pick 2π-offsets that best align the seed's
// phase evolution with a second echo. Ports the `if haskey(:phase2,:TEs)`
// branch of `seedcorrection!` verbatim, including the `(|off1| + |off2|)/100`
// tiebreaker that penalizes larger offsets (matters when TE1 == 2*TE2).
template <typename T>
inline void seed_correction_multiecho(T* wrapped, std::ptrdiff_t vox, const T* phase2, T te1, T te2) {
    constexpr T two_pi = static_cast<T>(2.0L * 3.141592653589793238462643383279502884L);
    T best = std::numeric_limits<T>::infinity();
    int chosen_off1 = 0;
    for (int off1 = -2; off1 <= 2; ++off1) {
        for (int off2 = -1; off2 <= 1; ++off2) {
            T diff = std::abs((wrapped[vox] + two_pi * static_cast<T>(off1)) / te1 -
                              (phase2[vox] + two_pi * static_cast<T>(off2)) / te2);
            diff += static_cast<T>(std::abs(off1) + std::abs(off2)) / T(100);
            if (diff < best) {
                best = diff;
                chosen_off1 = off1;
            }
        }
    }
    wrapped[vox] += two_pi * static_cast<T>(chosen_off1);
}

}  // namespace romeo

#endif
