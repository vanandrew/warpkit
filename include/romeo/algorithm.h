#ifndef ROMEO_ALGORITHM_H
#define ROMEO_ALGORITHM_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "romeo/priority_queue.h"
#include "romeo/seed.h"
#include "romeo/utility.h"

namespace romeo {

// ----------------------------------------------------------------------------
// Edge index math
//
// The weights array is (3, nx, ny, nz) in column-major order. Flat index
// `edge_index(vox, dim) = dim + 3*vox` with dim ∈ {0,1,2} (x/y/z forward
// neighbor) and vox 0-based linear. Mirrors `getedgeindex(leftvoxel, dim)`
// from ROMEO.jl src/algorithm.jl (translated from Julia 1-based).
// ----------------------------------------------------------------------------

inline int edge_dim(std::ptrdiff_t edge) { return static_cast<int>(edge % 3); }
inline std::ptrdiff_t edge_first_vox(std::ptrdiff_t edge) { return edge / 3; }
inline std::ptrdiff_t edge_index(std::ptrdiff_t vox, int dim) {
    return static_cast<std::ptrdiff_t>(dim) + 3 * vox;
}

// ----------------------------------------------------------------------------
// Voxel-to-voxel unwrap via 2π snapping. Ports `unwrapvoxel` / `unwrapedge!`.
// ----------------------------------------------------------------------------

template <typename T>
inline T unwrap_voxel(T new_v, T old_v) {
    constexpr T two_pi = static_cast<T>(2.0L * 3.141592653589793238462643383279502884L);
    return new_v - two_pi * std::round((new_v - old_v) / two_pi);
}

// Unwrap `newvox` relative to `oldvox`. If the voxel on the "other side" of
// `oldvox` (call it `oo = 2*oldvox - newvox`) is already visited, use the
// oldvox→oo phase step as an extra bias (clipped to ±wrap_addition) when
// predicting newvox's phase. Mirrors ROMEO.jl src/algorithm.jl.
template <typename T>
inline void unwrap_edge(T* wrapped, std::ptrdiff_t oldvox, std::ptrdiff_t newvox, const std::uint8_t* visited,
                        std::ptrdiff_t n, T wrap_addition) {
    const std::ptrdiff_t oo = 2 * oldvox - newvox;
    T d = T(0);
    if (oo >= 0 && oo < n && visited[oo] != 0) {
        T v = wrapped[oldvox] - wrapped[oo];
        if (v < -wrap_addition) {
            d = -wrap_addition;
        } else if (v > wrap_addition) {
            d = wrap_addition;
        } else {
            d = v;
        }
    }
    wrapped[newvox] = unwrap_voxel(wrapped[newvox], wrapped[oldvox] + d);
}

// ----------------------------------------------------------------------------
// grow_region_unwrap — the MST of quality-weighted edges. Ports
// ROMEO.jl src/algorithm.jl minus the merge_regions / correct_regions
// tail (warpkit never enables those).
// ----------------------------------------------------------------------------

template <typename T>
struct GrowRegionContext {
    T* wrapped;
    std::size_t nx, ny, nz;
    std::ptrdiff_t n;              // nx*ny*nz
    std::ptrdiff_t strides[3];     // {1, nx, nx*ny}
    const std::uint8_t* weights;   // (3, nx, ny, nz)
    std::uint8_t* visited;         // (nx, ny, nz), 0 = unvisited
    T wrap_addition;
    // Seed-correction context. When `phase2` is non-null the multi-echo branch
    // of seed_correction fires; otherwise the single-echo rem2pi path fires.
    const T* phase2;
    T te1;
    T te2;
};

namespace detail {

template <typename T>
inline bool in_bounds(const GrowRegionContext<T>& ctx, std::ptrdiff_t vox) {
    return vox >= 0 && vox < ctx.n;
}

template <typename T>
inline bool not_visited(const GrowRegionContext<T>& ctx, std::ptrdiff_t vox) {
    return in_bounds(ctx, vox) && ctx.visited[vox] == 0;
}

// Mirrors `getnewedge(v, notvisited, stridelist, i)` — returns the edge index
// of a neighbor in direction i ∈ {0..5} (0=−x, 1=+x, 2=−y, 3=+y, 4=−z, 5=+z),
// or -1 if that neighbor is out of bounds or already visited.
//
// Julia uses 1..6 with `div(i+1,2)` for the dim and `iseven(i)` for direction.
// In 0-based indexing here: dim = i/2, forward = (i & 1).
template <typename T>
inline std::ptrdiff_t get_new_edge(const GrowRegionContext<T>& ctx, std::ptrdiff_t vox, int i) {
    const int dim = i / 2;
    const std::ptrdiff_t step = ctx.strides[dim];
    if (i & 1) {  // forward
        if (not_visited(ctx, vox + step)) return edge_index(vox, dim);
    } else {  // backward
        if (not_visited(ctx, vox - step)) return edge_index(vox - step, dim);
    }
    return -1;
}

// Mirrors `getvoxelsfromedge(edge, visited, stridelist)` — returns
// (oldvox, newvox), where oldvox is the already-visited endpoint.
template <typename T>
inline std::pair<std::ptrdiff_t, std::ptrdiff_t> get_voxels_from_edge(const GrowRegionContext<T>& ctx,
                                                                     std::ptrdiff_t edge) {
    const int dim = edge_dim(edge);
    const std::ptrdiff_t vox = edge_first_vox(edge);
    const std::ptrdiff_t neighbor = vox + ctx.strides[dim];
    if (ctx.visited[neighbor] == 0) {
        return {vox, neighbor};
    }
    return {neighbor, vox};
}

}  // namespace detail

template <typename T>
inline void grow_region_unwrap(GrowRegionContext<T>& ctx, int maxseeds) {
    if (maxseeds > 255) maxseeds = 255;  // Julia: stored in UInt8 → hard cap.

    // Seed queue: voxels keyed by the sum of their three outgoing edge
    // weights, where weight=0 (invalid edge) counts as 255. Julia:
    //   sum([w == 0 ? UInt8(255) : w for w in weights]; dims=1)
    BucketQueue<std::ptrdiff_t> seed_queue(3 * NBINS);
    for (std::ptrdiff_t i = 0; i < ctx.n; ++i) {
        int s = 0;
        for (int d = 0; d < 3; ++d) {
            const int w = ctx.weights[edge_index(i, d)];
            s += (w == 0) ? 255 : w;
        }
        seed_queue.enqueue(i, s);
    }

    BucketQueue<std::ptrdiff_t> pqueue(NBINS);
    std::vector<std::ptrdiff_t> seeds;
    int new_seed_thresh = NBINS;  // Julia's starting value — strictly above every real bin.

    auto add_seed = [&]() -> int {
        // findseed!: drain the seed queue until we hit an unvisited voxel.
        std::ptrdiff_t seed = -1;
        while (!seed_queue.empty()) {
            std::ptrdiff_t cand = seed_queue.dequeue();
            if (ctx.visited[cand] == 0) {
                seed = cand;
                break;
            }
        }
        if (seed < 0) return 255;  // sentinel from Julia: no seeds left.

        // Enqueue the 6 outgoing edges that touch not-yet-visited neighbors.
        for (int i = 0; i < 6; ++i) {
            std::ptrdiff_t e = detail::get_new_edge(ctx, seed, i);
            if (e >= 0 && ctx.weights[e] > 0) {
                pqueue.enqueue(e, ctx.weights[e]);
            }
        }

        // Snap the seed's phase. Multi-echo if phase2/TEs are provided.
        if (ctx.phase2 != nullptr) {
            seed_correction_multiecho(ctx.wrapped, seed, ctx.phase2, ctx.te1, ctx.te2);
        } else {
            seed_correction_single(ctx.wrapped, seed);
        }

        seeds.push_back(seed);
        ctx.visited[seed] = static_cast<std::uint8_t>(seeds.size());

        // New-seed threshold. Julia:
        //   seed_weights = weights[getedgeindex.(seed, 1:3)]
        //   NBINS - div(NBINS - sum(seed_weights)/3, 2)
        // Using raw weights here (no 0→255 substitution), and Julia's sum/3 is
        // a Float64; div(Float,2) truncates toward zero.
        int seed_sum = 0;
        for (int d = 0; d < 3; ++d) seed_sum += ctx.weights[edge_index(seed, d)];
        const double avg = static_cast<double>(seed_sum) / 3.0;
        const double thresh = static_cast<double>(NBINS) - std::trunc((static_cast<double>(NBINS) - avg) / 2.0);
        return static_cast<int>(thresh);
    };

    // Initial seed — grow_region_unwrap is called with an empty pqueue from
    // unwrap_3d, so we always take this branch. Phase 4's 4D temporal path
    // will pre-populate pqueue and skip it.
    if (pqueue.empty()) {
        new_seed_thresh = add_seed();
    }

    // MST loop.
    while (!pqueue.empty()) {
        if (static_cast<int>(seeds.size()) < maxseeds && pqueue.min_bin() > new_seed_thresh) {
            new_seed_thresh = add_seed();
        }
        const std::ptrdiff_t edge = pqueue.dequeue();
        const auto [oldvox, newvox] = detail::get_voxels_from_edge(ctx, edge);
        if (ctx.visited[newvox] != 0) continue;

        unwrap_edge(ctx.wrapped, oldvox, newvox, ctx.visited, ctx.n, ctx.wrap_addition);
        ctx.visited[newvox] = ctx.visited[oldvox];

        for (int i = 0; i < 6; ++i) {
            std::ptrdiff_t e = detail::get_new_edge(ctx, newvox, i);
            if (e >= 0 && ctx.weights[e] > 0) {
                pqueue.enqueue(e, ctx.weights[e]);
            }
        }
    }
}

}  // namespace romeo

#endif
