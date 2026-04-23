# C++ port of ROMEO — implementation plan

## Scope

### Public surface (pybind11-exposed)

Only two entry points survive. Parameter shapes match the current Julia-backed
`include/romeo.h` so Python callers don't change.

1. `voxel_quality(phase4D, TEs, mag4D) → qmap3D`
2. `unwrap(phase3D_or_4D, mag, mask, weights="romeo", correct_global=True,
         maxseeds=1, TEs=None) → unwrapped`

The current code uses a `JuliaContext<T>` class — we'll rename to `Romeo<T>`
(or drop the class entirely in favor of free functions; TBD). We keep a
Python-side alias `JuliaContext = Romeo` for one release so [warpkit/unwrap.py](../warpkit/unwrap.py)
callers can migrate lazily.

### What we implement internally (needed by the two public entry points)

- Bucket priority queue (256 bins, UInt8 weights, dequeue from back)
- Weight calculation for the `:romeo` variant only — flags 1–4
  (phasecoherence, phasegradientcoherence, phaselinearity, magcoherence)
- Seed selection + seed correction (both phase2/TEs and single-echo branches)
- MST region growing (`grow_region_unwrap`), edge index math, `unwrap_edge`
- 3D unwrap with `correctglobal` (global 2π median shift inside mask)
- 4D unwrap orchestration: template-echo spatial unwrap + temporal unwrap of
  remaining echoes via `unwrap_voxel(current, reference * TEs[i]/TEs[ref])`
- voxelquality post-processing: sum over 3 edge dims, neighbor-add, `/6`

### What we **don't** port (dead weight for warpkit)

- `:bestpath` weights → `calculateweights_bestpath`, `getbestpathweight`, `getD`
- `:romeo2`/`:romeo3`/`:romeo4`/`:romeo6` and flags 5–6
  (`magweight`, `magweight2`)
- `unwrap_individual!` (`individual=True` path)
- `merge_regions!`, `correct_regions!`, `get_offset_count_sorted`
  (whole `region_handling.jl`)
- `getseededges`, `initqueue`, `getquality` (used only by
  `temporal_uncertain_unwrapping`, which warpkit never enables)
- `wrap_addition` parameter (always 0)
- All Julia/ROMEO build-time coupling (`cmake_aux/FindJulia.cmake`,
  `warpkit/julia.py`, ROMEO package install in [setup.py](../setup.py))

## Design decisions

- **Header-only** under `include/romeo/`. Templated on element type `T` (float/
  double) since the existing API templates `JuliaContext<T>`.
- **No new heavy deps.** Use raw pointers + explicit strides. No Eigen, xtensor,
  or std::mdspan (C++17 target, [CMakeLists.txt:9](../CMakeLists.txt#L9)).
- **Column-major (F-order) throughout**, matching `py::array::f_style` coming
  from Python and Julia's default.
- **NaN handling** must match ROMEO literally: NaN voxel → weight 0 (edge never
  enqueued); edge adjacent to a NaN voxel → weight 157 (the literal from
  [specialcases.jl](https://github.com/korbinian90/ROMEO.jl/blob/v1.0.0/test/specialcases.jl)).
- **Dequeue semantics**: pop from the back of each bin (Julia's `pop!`) — we
  mirror this for reproducibility rather than popping from the front.
- Public API parameter names stay identical. Internally use snake_case.

## File layout

```
include/romeo/
  volume_view.h       # Volume3D<T> / Volume4D<T> — ptr + strides + extents + accessors
  utility.h           # gamma_fold, NBINS = 256, rem2pi_nearest
  priority_queue.h    # BucketQueue<T>
  weights.h           # calculate_weights_romeo, getweight, rescale, NaN plumbing
  seed.h              # seed_selector, find_seed, seed_correction
  algorithm.h         # grow_region_unwrap, unwrap_edge, edge math
  unwrap.h            # unwrap_3d, unwrap_4d
  voxel_quality.h     # voxel_quality
src/
  warpkit.cpp         # unchanged pybind11 bindings except class name
include/romeo.h       # DELETED (replaced by include/romeo/*.h)
```

## Phased rollout

Each phase should leave the tree building and `uv run pytest tests/test_romeo.py`
passing (skipping tests that don't yet apply, with skip messages referring to
the next phase).

### Phase 1 — Remove Julia, add empty C++ scaffolding
*Goal: `uv sync --group dev` succeeds on a machine with **no Julia installed**; all tests skip.*

- Delete `cmake_aux/FindJulia.cmake`.
- Delete the `find_package(Julia)` + CIBUILDWHEEL Julia block from [CMakeLists.txt](../CMakeLists.txt).
- Delete the `Pkg.add(ROMEO)` execute_process block.
- Remove `PRIVATE Julia` from `target_link_libraries(warpkit_cpp ...)`.
- Delete the Julia preflight in [setup.py](../setup.py) (`subprocess.run(["julia", ...])` blocks).
- Delete [warpkit/julia.py](../warpkit/julia.py).
- In [.github/workflows/build.yml](../.github/workflows/build.yml) + [Dockerfile](../Dockerfile): strip all `julia-actions/setup-julia`, `/opt/julia/lib` ld.so.conf plumbing, `brew install julia`, `DYLD_LIBRARY_PATH`, `auditwheel repair --exclude libjulia.so.1`, and the matching `before-all` block in `[tool.cibuildwheel.*]` in [pyproject.toml](../pyproject.toml).
- Create `include/romeo/*.h` with declarations only; implementations just `throw std::logic_error("not implemented");`.
- Rewrite [src/warpkit.cpp](../src/warpkit.cpp) bindings to point at the new classes, keeping parameter names identical. Expose class as `Romeo`, plus a Python-level alias `JuliaContext = Romeo` in [warpkit/__init__.py](../warpkit/__init__.py) for one release.
- Delete `include/romeo.h`.
- **Verify**: repo builds on macOS/Linux with zero Julia-adjacent tooling; `pytest tests/test_romeo.py -v` shows 10 skipped (same as today).

### Phase 2 — Utility + priority queue + weights
*Goal: `test_weight_calc_literals` passes (remove its `xfail`).*

- Implement `gamma_fold`, `NBINS`, `rem2pi_nearest` in `romeo/utility.h`.
- Implement `BucketQueue<int>` in `romeo/priority_queue.h` — 256 bins, O(1) enqueue/dequeue, tracks `min` bin. Unit-tested via a one-off [warpkit_cpp.debug_bucket_queue] binding (kept private).
- Implement `calculate_weights_romeo<T>` in `romeo/weights.h` — only the `:romeo` flag path. Output is (3, D0, D1, D2) uint8.
  - NaN-adjacent edge = 157 literal (pre-rescale raw bin), NaN-voxel edges = 0.
  - `rescale(w)` = `max(round((1-w)*(NBINS-1)), 1)` for `w ∈ [0,1]`, else 0.
- Expose a `calculate_weights(phase, mag=None, TEs=None, phase2=None)` method on `Romeo<T>` so the 2 parametrized goldens in [tests/test_romeo.py](../tests/test_romeo.py) can drive it.
- **Verify**: drop `@pytest.mark.xfail` + `pytest.skip` from `test_weight_calc_literals`; it passes.

### Phase 3 — Seed + algorithm + 3D unwrap
*Goal: `test_unwrap_1d_literals` (5 cases) + `test_unwrap3d_property` pass.*

- `romeo/seed.h`: seed selection via a `BucketQueue<int>` of per-voxel weight sums; `seed_correction` with both branches (phase2/TEs multi-echo; single-echo `rem2pi_nearest`).
- `romeo/algorithm.h`: `grow_region_unwrap` with `maxseeds`. Port the edge-index helpers exactly. Skip the `merge_regions`/`correct_regions` tail.
- `romeo/unwrap.h`: `unwrap_3d` — calls weights, grow_region, then if `correct_global`: compute `2π * median(round(phase[mask ∧ isfinite] / (2π)))` and subtract from all voxels.
- Wire `romeo_unwrap3D` binding to `unwrap_3d`.
- **Verify**: 1D goldens pass (reshape to (N,1,1)); 3D property test passes on `Phase.nii`.

### Phase 4 — 4D unwrap
*Goal: `test_unwrap4d_property` passes.*

- `unwrap_4d`: spatial-unwrap the template echo (default 0) using `unwrap_3d`; for each other echo `e`, compute `ref = wrapped[..., iref] * (TEs[e]/TEs[iref])` and apply `unwrap_voxel(current, ref)` element-wise. `iref` = template-1 below, template+1 above.
- Wire `romeo_unwrap4D` binding.
- **Verify**: 4D property test passes.

### Phase 5 — voxelquality
*Goal: `test_voxelquality_behavior` passes.*

- `voxel_quality`: call `calculate_weights_romeo` with `rescale = identity`, sum across the 3 edge dims, then the neighbor-shift-add trick (`qmap[1:,:,:] += weights[0,:-1,:,:]`, etc.), divide by 6.
- For 4D input: template=0, p2ref=1, pull `mag[..., 0]` if provided.
- Wire `romeo_voxelquality` binding.
- **Verify**: voxelquality test passes.

### Phase 6 — Cleanup + release
*Goal: remove every trace of Julia.*

- Drop the `JuliaContext` Python alias from [warpkit/__init__.py](../warpkit/__init__.py).
- Update [warpkit/unwrap.py](../warpkit/unwrap.py): replace `JULIA` references with `Romeo()` or module-level function calls; remove `from .julia import JuliaContext`.
- Update [README.md](../README.md) to drop Julia install notes.
- Drop `JULIA_VERSION_TRACK` / cibuildwheel before-all blocks that still reference julia/delocate — the full `[tool.cibuildwheel.linux]` / `[tool.cibuildwheel.macos]` `before-all` go away.
- Remove the `TODO(romeo-port):` skip from [tests/test_romeo.py](../tests/test_romeo.py). All 10 tests run.

## Python-side impact

| File | Change |
|---|---|
| [warpkit/julia.py](../warpkit/julia.py) | delete after Phase 6 (or Phase 1 if we don't need the alias) |
| [warpkit/__init__.py](../warpkit/__init__.py) | export `Romeo` (new name); temp alias `JuliaContext = Romeo` during transition |
| [warpkit/unwrap.py](../warpkit/unwrap.py) | `JULIA = JuliaContext()` → `ROMEO = Romeo()`; rename `JULIA.romeo_*` call sites |
| [tests/test_romeo.py](../tests/test_romeo.py) | import `Romeo` instead of `JuliaContext`; drop module-level skip |

## Risks / open questions

1. **Numerical parity is unverifiable.** We can't diff against the Julia
   backend because it no longer builds. The bar is the property tests in
   [tests/test_romeo.py](../tests/test_romeo.py) (2π-modulo inside mask,
   finite outputs, qmap ∈ [0,1]) plus the literal goldens from
   [dsp_tests.jl](https://github.com/korbinian90/ROMEO.jl/blob/v1.0.0/test/dsp_tests.jl) and
   [specialcases.jl](https://github.com/korbinian90/ROMEO.jl/blob/v1.0.0/test/specialcases.jl). Once the port
   is stable, capture its own outputs as regression goldens in a follow-up
   (`.npy` snapshot of qmap + unwrapped for the test volume).
2. **`median` semantics.** Julia's `Statistics.median` on even-length samples
   returns the mean of the two middle values. `std::nth_element` + hand-written
   pick-two; don't use naive sort for the full `correct_global` path on large
   volumes.
3. **NaN propagation through weights.** Missing this silently produces wrong
   unwrapping near skull/background boundaries. `specialcases.jl` goldens cover
   the 1D case directly — add a 3D NaN-mask test as a follow-up.
4. **Edge enumeration order.** `grow_region_unwrap`'s 6-direction inner loop
   order affects which neighbor is reached first when weights tie; Julia's
   order (i=1..6 → ±x, ±y, ±z in a specific permutation) should be matched
   bit-for-bit to minimize surprise.
5. **`maxseeds>1` untested.** warpkit only uses `maxseeds=1`, so the seed
   threshold arithmetic is hard to validate. Acceptable — document and punt.
6. **Threading.** Current Julia path is single-threaded (global interpreter
   lock of Julia runtime forced it). C++ port could OpenMP the weight
   calculation and voxelquality summations trivially. Out of scope for first
   cut; track as a follow-up once correctness is nailed.

## Done-criteria

- `uv sync --group dev` succeeds on a machine with **no Julia, no Julia
  packages, no ROMEO registry install**.
- `uv run pytest tests/test_romeo.py` — 10 passed, 0 skipped, 0 xfailed.
- `docker build .` succeeds without fetching Julia.
- `grep -ri julia . --exclude-dir=.venv` returns
  zero hits (except the project-wide `JuliaContext` alias if still present).
- No regressions in the other tests (test_concurrency, test_model,
  test_utilities run green; test_distortion stays skipped — separate issue).
