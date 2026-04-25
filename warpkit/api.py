"""Typed Python entry points for the seven warpkit operations.

Every ``wk-*`` CLI tool is mirrored here as a typed Python function so
library callers (e.g. nipype interfaces, fmriprep) can drive warpkit without
shelling out or fabricating ``argv``. Each function takes keyword-only
arguments, raises :class:`ValueError` on validation problems, writes its
outputs to disk, and returns a frozen dataclass with the absolute paths of
the written NIfTIs.

Mapping CLI -> Python:

* ``wk-medic``              -> :func:`medic` -> :class:`MedicResult`
* ``wk-unwrap-phase``       -> :func:`unwrap_phase` -> :class:`UnwrapPhaseResult`
* ``wk-compute-fieldmap``   -> :func:`compute_fieldmap` -> :class:`ComputeFieldmapResult`
* ``wk-apply-warp``         -> :func:`apply_warp` -> :class:`ApplyWarpResult`
* ``wk-convert-warp``       -> :func:`convert_warp` -> :class:`ConvertWarpResult`
* ``wk-convert-fieldmap``   -> :func:`convert_fieldmap` -> :class:`ConvertFieldmapResult`
* ``wk-compute-jacobian``   -> :func:`compute_jacobian` -> :class:`ComputeJacobianResult`

The CLI flag-name → Python kwarg mapping is the obvious dash-to-underscore
transform; the only difference is ``--TEs`` (kept for MR convention) →
``tes`` (lowercase, per repo style).
"""

from .scripts.apply_warp import ApplyWarpResult, apply_warp
from .scripts.compute_fieldmap import ComputeFieldmapResult, compute_fieldmap
from .scripts.compute_jacobian import ComputeJacobianResult, compute_jacobian
from .scripts.convert_fieldmap import ConvertFieldmapResult, convert_fieldmap
from .scripts.convert_warp import ConvertWarpResult, convert_warp
from .scripts.medic import MedicResult, medic
from .scripts.unwrap_phase import UnwrapPhaseResult, unwrap_phase

__all__ = [
    "ApplyWarpResult",
    "ComputeFieldmapResult",
    "ComputeJacobianResult",
    "ConvertFieldmapResult",
    "ConvertWarpResult",
    "MedicResult",
    "UnwrapPhaseResult",
    "apply_warp",
    "compute_fieldmap",
    "compute_jacobian",
    "convert_fieldmap",
    "convert_warp",
    "medic",
    "unwrap_phase",
]
