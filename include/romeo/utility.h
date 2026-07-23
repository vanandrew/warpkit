#ifndef ROMEO_UTILITY_H
#define ROMEO_UTILITY_H

#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>

namespace romeo {

// Number of weight bins used by the bucket priority queue.
// Matches ROMEO.jl's `const NBINS = 256`.
constexpr int NBINS = 256;

// Fold x into [-π, π] when only one wrap can have occurred.
// Ports `γ(x)` from ROMEO.jl src/utility.jl.
template <std::floating_point T>
inline T gamma_fold(T x) {
    constexpr T pi = std::numbers::pi_v<T>;
    constexpr T two_pi = T(2) * pi;
    if (x < -pi) return x + two_pi;
    if (x > pi) return x - two_pi;
    return x;
}

// Julia's rem2pi(x, RoundNearest) — fold x into (-π, π], no fast-path.
template <std::floating_point T>
inline T rem2pi_nearest(T x) {
    constexpr T two_pi = T(2) * std::numbers::pi_v<T>;
    return x - two_pi * std::round(x / two_pi);
}

}  // namespace romeo

#endif
