"""Shared output handling for the MEDIC field-map scripts.

``wk-medic`` and ``wk-compute-fieldmap`` write the same three NIfTIs from a
prefix — native-space field map, displacement maps, undistorted-space field
map — and return the same triple of resolved paths. This module hosts the
result type and the writer they share.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import nibabel as nib


@dataclass(frozen=True, slots=True)
class MedicResult:
    """Absolute paths of the three NIfTIs written by the MEDIC field-map
    scripts: native-space field map, displacement maps, and undistorted-space
    field map."""

    fieldmap_native: Path
    displacement_map: Path
    fieldmap: Path


def write_medic_outputs(
    out_prefix: PathLike[str] | str,
    fieldmap_native: nib.Nifti1Image,
    displacement_map: nib.Nifti1Image,
    fieldmap: nib.Nifti1Image,
) -> MedicResult:
    """Write the three MEDIC NIfTIs under ``out_prefix`` and return their
    resolved paths.

    Writes ``<prefix>_fieldmaps_native.nii``, ``<prefix>_displacementmaps.nii``
    and ``<prefix>_fieldmaps.nii``.
    """
    out_prefix_str = str(out_prefix)
    print("Saving field maps and displacement maps to file...")
    fmap_native_path = Path(f"{out_prefix_str}_fieldmaps_native.nii").resolve()
    dmap_path = Path(f"{out_prefix_str}_displacementmaps.nii").resolve()
    fmap_path = Path(f"{out_prefix_str}_fieldmaps.nii").resolve()
    fieldmap_native.to_filename(str(fmap_native_path))
    displacement_map.to_filename(str(dmap_path))
    fieldmap.to_filename(str(fmap_path))
    print("Done.")
    return MedicResult(
        fieldmap_native=fmap_native_path,
        displacement_map=dmap_path,
        fieldmap=fmap_path,
    )
