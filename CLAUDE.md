# CLAUDE.md

Guidance for Claude when working in this repo.

## What warpkit is

A Python library for neuroimaging transforms, focused on MEDIC (Multi-Echo
DIstortion Correction). Phase unwrapping uses a self-contained C++17 port of
[ROMEO](https://github.com/korbinian90/ROMEO) under `include/romeo/` —
**no Julia runtime is involved**; do not reintroduce one (no `julia.py`, no
conda recipe, no `FindJulia.cmake`).

Pre-print: <https://www.biorxiv.org/content/10.1101/2023.11.28.568744v1>.

## Layout

- `warpkit/` — Python package. Public entry points: `warpkit.distortion.medic`,
  `warpkit.utilities.*`. CLI scripts in `warpkit/scripts/` are auto-discovered
  by `setup.py` and exposed as `medic` and `extract_field_from_maps`.
- `warpkit/warpkit_cpp.pyi` + `warpkit/py.typed` — type info for the compiled
  extension, shipped via `MANIFEST.in` and `[tool.setuptools.package-data]`.
  Regenerate the stub after pybind11 binding changes via the wrapper script —
  it runs pybind11-stubgen and rewrites the `numpy.bool` → `numpy.bool_`
  artifact that stubgen 2.5.x emits:
  ```bash
  scripts/regen-stub.sh
  ```
- `src/warpkit.cpp` — pybind11 bindings. C++ headers (and the vendored ROMEO
  port) live under `include/`. CMake fetches and statically links ITK 5.4 and
  pybind11 3.0 — first build is slow (~1m+), subsequent rebuilds are quick.
- `tests/` — pytest tests. `conftest.py` loads MEDIC sample data from
  `tests/data/test_data/`. ROMEO port-validation goldens in `tests/data/romeo/`.

## Build & dev workflow

```bash
# editable install (rebuilds the CMake extension on demand)
uv sync --group dev --config-setting editable_mode=strict

# tests
uv run pytest -q

# lint + types (matches pre-commit; never bypass with --no-verify)
uv run --with ruff ruff check
uv run --with ruff ruff format
uv run --with pyright pyright

# regenerate the .pyi after touching src/warpkit.cpp
scripts/regen-stub.sh

# install pre-commit hooks (incl. the commit-msg gitmoji check)
uv run pre-commit install
```

`default_install_hook_types` in `.pre-commit-config.yaml` ensures
`pre-commit install` registers both the `pre-commit` and `commit-msg` stages.

## Code conventions

- Python ≥ 3.11. Use modern syntax: `X | None`, `list[T]`, `tuple[T, ...]`,
  `Callable` / `Iterator` / `Sequence` from `collections.abc`. Ruff's `UP`
  rules enforce this.
- Lowercase snake_case **even for MR-physics symbols**: use `tes`, `te0`,
  `te1`, `tr_in_sec`, `b0` — never `TEs`, `TE0`, etc. Ruff `N803`/`N806` is
  selected with no per-file ignore.
- pybind11 bindings exposed to Python are also lowercase
  (`romeo_unwrap3d`, `romeo_unwrap4d`, kwarg `tes`). The C++ method names in
  `include/romeo/` keep their `3D`/`4D` casing — only the Python-visible
  binding name is constrained.
- The CLI keeps `--TEs` as the user-facing flag for MR convention, but the
  argparse `dest=` is `tes` so internal Python is lowercase.
- Don't add blanket `# type: ignore` on imports or function bodies. Fix the
  type, narrow with `cast`, or pin the ignore to a specific rule
  (`# pyright: ignore[reportAttributeAccessIssue]`).

## Commit and PR style

Use the [gitmoji](https://gitmoji.dev/) standard. Subject line:

```
:emoji_shortcode: <Imperative subject under ~70 chars>
```

Use the colon-shortcode form (`:sparkles:`), not the unicode glyph (`✨`).
Common shortcodes for this repo:

| shortcode             | when                                |
| --------------------- | ----------------------------------- |
| `:sparkles:`          | new feature                         |
| `:bug:`               | bug fix                             |
| `:recycle:`           | refactor with no behavior change    |
| `:zap:`               | perf                                |
| `:fire:`              | remove code or files                |
| `:white_check_mark:`  | tests                               |
| `:memo:`              | docs                                |
| `:label:`             | typing / type stubs                 |
| `:wrench:`            | config / build / CI                 |
| `:arrow_up:`          | bump dependency versions            |
| `:rocket:`            | release / deploy                    |

The `commit-msg` hook in `.pre-commit-config.yaml` enforces this on every
commit; merge / revert / fixup / squash / amend commits are exempt.

PR titles follow the same convention. Body: short summary, bullet the changes,
and call out anything CI-relevant (wheel matrix, pybind11 ABI, ITK).

## CI specifics

GitHub Actions builds wheels for Python 3.11–3.14 on `ubuntu-latest` and
`macos-latest` via cibuildwheel. `pyproject.toml`'s `[tool.cibuildwheel]`
skips `*musllinux*` and free-threaded builds (`cp313t-*`, `cp314t-*`);
re-enabling free-threaded support requires auditing the pybind11 + ITK code
paths for the no-GIL ABI. PyPI publish and the GHCR Docker image only run on
a published GitHub release.
