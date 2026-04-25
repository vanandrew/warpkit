# ROMEO C++ port

Header-only implementation of the algorithms behind ROMEO.jl's `unwrap` and
`voxelquality`. Public entry points are bundled into `Romeo<T>` (see
`romeo.h`).

## Provenance

Ported from [ROMEO.jl](https://github.com/korbinian90/ROMEO.jl) at **v1.0.0**
(commit `9faef5bb8a9d8b251822f618a26df3d4337d2891`). The [LICENSE](LICENSE)
file is the upstream MIT license, reproduced here per its terms. Individual
header comments point at the corresponding Julia source files (e.g.
`ROMEO.jl src/weights.jl`) so the algorithmic provenance stays obvious.

The port deliberately omits anything warpkit doesn't exercise:

- weight presets other than `:romeo`
- the `:bestpath` weight path
- `individual=true` on the 4D unwrap
- `merge_regions` / `correct_regions`
- `temporal_uncertain_unwrapping`
- flags 5–6 (`magweight`, `magweight2`)

A couple of small behavior choices worth knowing:

- `phase_linearity_triplet` adopts the post-v1.0.0 ROMEO master NaN guard
  (triplet collapses to `0.5` rather than propagating NaN).
- The bucket priority queue matches Julia's LIFO-within-bin dequeue order so
  tie-breaking across voxels is deterministic.
