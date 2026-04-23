#ifndef ROMEO_UTILITY_H
#define ROMEO_UTILITY_H

#include <cmath>
#include <cstddef>

namespace romeo {

// Number of weight bins used by the bucket priority queue.
// Matches ROMEO.jl's `const NBINS = 256`.
constexpr int NBINS = 256;

// Fold x into [-π, π] when only one wrap can have occurred.
// Ports `γ(x)` from ROMEO.jl src/utility.jl.
template <typename T>
inline T gamma_fold(T x) {
    constexpr T two_pi = static_cast<T>(2.0L * 3.141592653589793238462643383279502884L);
    constexpr T pi = static_cast<T>(3.141592653589793238462643383279502884L);
    if (x < -pi) return x + two_pi;
    if (x > pi) return x - two_pi;
    return x;
}

// Julia's rem2pi(x, RoundNearest) — fold x into (-π, π], no fast-path.
template <typename T>
inline T rem2pi_nearest(T x) {
    constexpr T two_pi = static_cast<T>(2.0L * 3.141592653589793238462643383279502884L);
    return x - two_pi * std::round(x / two_pi);
}

}  // namespace romeo

#endif
