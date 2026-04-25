# warpkit

[![Build](https://github.com/vanandrew/warpkit/actions/workflows/build.yml/badge.svg)](https://github.com/vanandrew/warpkit/actions)
[![PyPI](https://img.shields.io/pypi/v/warpkit)](https://pypi.org/project/warpkit/)
[![docker](https://ghcr-badge.egpl.dev/vanandrew/warpkit/latest_tag?trim=major&label=ghcr&nbsp;latest)](https://github.com/vanandrew/warpkit/pkgs/container/warpkit)
[![codecov](https://codecov.io/gh/vanandrew/warpkit/graph/badge.svg?token=S6ZZKOAF8V)](https://codecov.io/gh/vanandrew/warpkit)

A Python library for neuroimaging transforms, focused on the Multi-Echo DIstortion Correction (MEDIC) algorithm. The pre-print is available at <https://www.biorxiv.org/content/10.1101/2023.11.28.568744v1>.

The phase-unwrapping core is a self-contained C++17 port of [ROMEO](https://github.com/korbinian90/ROMEO) — there is no Julia runtime to install, and binary wheels ship with the ITK pieces statically linked. If you used an older release of warpkit that required `julia` on `PATH`, that step is gone.

## Installation

### From PyPI (recommended)

```bash
pip install warpkit
```

Pre-built wheels are published for Linux (x86_64) and macOS (universal2). If `pip` falls back to a source build and fails, please open an issue with the output of `pip install warpkit -v`.

### Docker

```bash
docker run -it --rm ghcr.io/vanandrew/warpkit:latest --help
```

The image's entrypoint is the `medic` CLI.

### From source

You need a C++17 compiler and CMake ≥ 3.24. Everything else (Python, build deps, ITK) is resolved by [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vanandrew/warpkit.git
cd warpkit
uv sync --group dev --config-setting editable_mode=strict
```

`editable_mode=strict` is the [recommended setuptools editable layout](https://setuptools.pypa.io/en/latest/userguide/development_mode.html#strict-editable-installs) — it avoids the import-path pitfalls of the default editable install. Append `-v` to `uv sync` if a build fails and include the full log when reporting the issue.

## What is MEDIC?

MEDIC takes ME-EPI phase data:

![phase](notes/phase.png)

and turns it into a per-frame field map:

![field map](notes/fmap.png)

You can then use these field maps to distortion-correct your data.

## Using MEDIC

`warpkit` is meant to be integrated into a larger neuroimaging pipeline. The Python entry point:

```python
import nibabel as nib
from warpkit.distortion import medic
from warpkit.utilities import displacement_map_to_field

# each list entry is a different echo
phases = [nib.load(p) for p in phases_paths]
magnitudes = [nib.load(p) for p in magnitude_paths]
TEs = [TE1, TE2, ...]                     # milliseconds
total_readout_time = ...                  # seconds
phase_encoding_direction = ...            # one of i, j, k, i-, j-, k-, x, y, z, x-, y-, z-

field_maps_native, displacement_maps, field_maps = medic(
    phases, magnitudes, TEs, total_readout_time, phase_encoding_direction
)
```

Returns are `nibabel.Nifti1Image` objects:

- `field_maps_native` — field maps in distorted space (Hz). Mostly useful for debugging.
- `displacement_maps` — displacement in undistorted space (mm). Convert with `displacement_map_to_field` before applying.
- `field_maps` — field maps in undistorted space (Hz). Equivalent to topup/fugue output, but framewise.

To convert a displacement map to a per-frame warp field for your tool of choice:

```python
displacement_maps.to_filename("/path/to/save.nii.gz")

displacement_field = displacement_map_to_field(
    displacement_maps, axis="y", format="itk", frame=0
)
```

`format` selects the output convention. Apply with the matching tool:

| `format` | Apply with                 | Notes                                                                  |
| -------- | -------------------------- | ---------------------------------------------------------------------- |
| `itk`    | `warpkit.utilities.resample_image` | warpkit's internal format.                                     |
| `ants`   | `antsApplyTransforms`      | ANTs uses ITK underneath, but its warp convention differs from ours.   |
| `afni`   | `3dNwarpApply`             |                                                                        |
| `fsl`    | `applywarp`                | If you'd rather use `fugue`, feed it `field_maps` instead.             |

### CLI

A `medic` script is installed on `PATH`. Acquisition parameters can come either from BIDS sidecars or from the command line — pick one.

From BIDS sidecars:

```bash
medic \
  --magnitude mag_e1.nii.gz mag_e2.nii.gz mag_e3.nii.gz \
  --phase phase_e1.nii.gz phase_e2.nii.gz phase_e3.nii.gz \
  --metadata mag_e1.json mag_e2.json mag_e3.json \
  --out-prefix sub-01_run-01
```

`EchoTime` is read per file (seconds → converted to ms internally); `TotalReadoutTime` and `PhaseEncodingDirection` are taken from the first sidecar.

Or by passing acquisition parameters directly:

```bash
medic \
  --magnitude mag_e1.nii.gz mag_e2.nii.gz mag_e3.nii.gz \
  --phase phase_e1.nii.gz phase_e2.nii.gz phase_e3.nii.gz \
  --TEs 14.2 38.93 63.66 \
  --total-readout-time 0.0501 \
  --phase-encoding-direction j- \
  --out-prefix sub-01_run-01
```

`--TEs` is in **milliseconds**, `--total-readout-time` in **seconds**, and `--phase-encoding-direction` is one of `i, j, k, i-, j-, k-, x, y, z, x-, y-, z-`.

Run `medic --help` for the full option list (noise-frame trimming, CPU count, debug mode, etc.).

## Authors

Vahdeta Suljic &lt;suljic@wustl.edu&gt;, Andrew Van &lt;vanandrew@wustl.edu&gt;
