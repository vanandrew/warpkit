# Vendored ROMEO source (reference only)

This directory contains a snapshot of the [ROMEO.jl](https://github.com/korbinian90/ROMEO.jl)
Julia package, used here as reference material while porting its algorithms to C++.

The Julia sources in `src/` are **not compiled or linked** by warpkit's build.
They exist so we can cross-check the C++ port against the original implementation.

## Origin

- Upstream: https://github.com/korbinian90/ROMEO.jl
- Version: v1.0.0
- Commit: 9faef5bb8a9d8b251822f618a26df3d4337d2891
- Retrieved: 2026-04-22
- License: MIT (see `LICENSE`)

## What we pulled

- `src/` — the full Julia module. Every file in `src/` is `include`d transitively
  from `src/ROMEO.jl`, so all nine are needed to read the algorithm end-to-end.
- `LICENSE` — required by MIT for redistribution.
- `README.md` — upstream docs, kept unmodified for faithful reference.
- `test/data/small/{Mag,Phase}.nii` — upstream's synthetic 4-D volume
  (51×51×41×3, float32, phase in [-π, π]). Used by `tests/test_romeo.py` as the
  test corpus for the C++ port. Replaces the unreachable `wustl.box.com` dataset
  the old `tests/data/download_bids_testdata.sh` script pointed at.
- `test/{dsp_tests,features,specialcases,mri,voxelquality}.jl` — reference only,
  **not executed**. Source of the numeric literals and property assertions the
  Python tests mirror.

## What we rely on from the port's perspective

The C++ side calls exactly three entry points (see `include/romeo.h`):

1. `voxelquality(phase; TEs, mag)` — defined in `src/voxelquality.jl`.
2. `unwrap(phase; weights, mag, mask, correctglobal, maxseeds, merge_regions, correct_regions)`
   — 3D overload, defined in `src/unwrapping.jl`.
3. `unwrap(phase; TEs, weights, mag, mask, correctglobal, maxseeds, merge_regions, correct_regions)`
   — 4D (multi-echo) overload, also in `src/unwrapping.jl`.

External Julia dependencies (per upstream `Project.toml`) are `Statistics` (Julia
stdlib) and `StatsBase` (for e.g. `median`) — both need replacements in C++.

## Updating

If we ever need to re-sync from upstream (e.g. to pick up a bug fix we want to
mirror), refresh the commit SHA above and re-copy `src/`, `LICENSE`, and
`README.md`. Do not modify files in `src/` — keep local changes in the C++ port
instead.
