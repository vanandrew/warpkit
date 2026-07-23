"""Orientation x phase-encoding-direction correctness.

MEDIC (and the correction it produces) is a physical operation: the same field
map, phase-encode direction, and image must yield the *same physical* correction
no matter which of the 48 valid voxel orientations the data happens to be stored
in on disk — provided the phase-encoding-direction (PED) string is relabelled to
match each storage orientation, exactly as BIDS requires.

The suite therefore takes RAS as the golden reference (the orientation warpkit
was validated against) and asserts that every other orientation reduces to the
*same* physical result. Because RAS is correct, this equivariance is equivalent
to correctness for every orientation.

Two absolute anchors in RAS pin the sign/magnitude so the sweep can't pass by
being uniformly wrong, and an oracle-guard test independently checks that the
PED-remap helper the sweep relies on really does track the intended physical
axis.

See ``tests/test_utilities.py`` for the per-function (Hz<->mm, map<->field)
convention tests; this file is the end-to-end orientation behaviour.
"""

import itertools
from collections.abc import Sequence
from typing import cast

import nibabel as nib
import numpy as np
import pytest
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from numpy.testing import assert_allclose
from warpkit.utilities import (
    AXIS_MAP,
    displacement_map_to_field,
    field_maps_to_displacement_maps,
    reconstruct_displacement_and_field_maps,
    resample_image,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# The three anatomical axis pairs (positive, negative endpoint per world axis).
_AXIS_PAIRS = (("R", "L"), ("A", "P"), ("S", "I"))


def _all_axcodes() -> list[tuple[str, str, str]]:
    """Every valid voxel orientation: the 6 axis permutations x 8 sign
    combinations = 48 axcode triples (RAS, LAS, PSR, ...)."""
    codes = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((0, 1), repeat=3):
            codes.append(tuple(_AXIS_PAIRS[perm[d]][signs[d]] for d in range(3)))
    return codes


ALL_AXCODES = _all_axcodes()

# base phase-encode directions in RAS, keyed by the world axis they point along
# (+R = world 0, +A = world 1, +S = world 2).
_BASE_PEDS = {0: "i", 1: "j", 2: "k"}


def _reorient(img: nib.Nifti1Image, axcodes: tuple[str, str, str]) -> nib.Nifti1Image:
    """Relabel ``img`` into the target voxel orientation (lossless: physical
    content unchanged, only the array layout + affine)."""
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(axcodes))
    # ornt_transform returns an ndarray; as_reoriented is typed for the nested
    # Sequence form nibabel also accepts.
    return img.as_reoriented(cast(Sequence[Sequence[int]], transform))


def _ped_for_axcodes(axcodes: tuple[str, str, str], base_world_axis: int) -> str:
    """The PED string that, in ``axcodes`` storage, points along the same
    physical direction as the positive ``base_world_axis`` does in RAS.

    ``axcodes2ornt(axcodes)[q] = (w, s)`` means voxel axis q maps to world axis
    w with sign s; we want the voxel axis carrying ``base_world_axis``, and a
    trailing ``-`` when increasing that index runs *against* the world axis.
    """
    ornt = axcodes2ornt(axcodes)
    for voxel_axis in range(3):
        world_axis, sign = ornt[voxel_axis]
        if int(world_axis) == base_world_axis:
            letter = "ijk"[voxel_axis]
            return letter if sign > 0 else letter + "-"
    raise AssertionError(f"no voxel axis maps to world axis {base_world_axis}")


# synthetic acquisition (RAS): anisotropic voxels + a non-trivial origin so the
# sweep is sensitive to axis-size and translation bookkeeping, an asymmetric
# smooth field map (Hz), and an off-centre blob to correct.
_SHAPE = (11, 13, 15)
_AFFINE = np.diag([2.0, 3.0, 4.0, 1.0])
_AFFINE[:3, 3] = [5.0, -7.0, 11.0]
_TRT = 0.04


def _ras_inputs() -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    ii, jj, kk = np.indices(_SHAPE)
    fmap = (
        20.0 * np.sin(2 * np.pi * (ii / _SHAPE[0] + jj / _SHAPE[1] + kk / _SHAPE[2]))
    ).astype(np.float32)
    centre = np.array([4.0, 8.0, 6.0])
    blob = np.exp(
        -(
            ((ii - centre[0]) / 3) ** 2
            + ((jj - centre[1]) / 2.5) ** 2
            + ((kk - centre[2]) / 3.5) ** 2
        )
    ).astype(np.float32)
    fmap_img = nib.Nifti1Image(fmap[..., None], _AFFINE)  # 4D (single frame)
    moving_img = nib.Nifti1Image(blob, _AFFINE)
    return fmap_img, moving_img


def _corrected(
    fmap: nib.Nifti1Image, moving: nib.Nifti1Image, ped: str
) -> nib.Nifti1Image:
    """Full correction pipeline: reconstruct the displacement map (with the
    C++ inversion) then apply it to ``moving`` — exactly what medic + apply-warp
    do, minus the ROMEO unwrap (which is per-voxel and orientation-agnostic)."""
    dmaps, _ = reconstruct_displacement_and_field_maps(fmap, _TRT, ped)
    field = displacement_map_to_field(dmaps, axis=ped, format="itk", frame=0)
    return resample_image(moving, moving, field)


@pytest.fixture(scope="module")
def ras_references() -> dict[int, np.ndarray]:
    """The corrected image in RAS for each base PE axis — the golden result
    every orientation must reproduce."""
    fmap, moving = _ras_inputs()
    return {
        world_axis: _corrected(fmap, moving, ped).get_fdata()
        for world_axis, ped in _BASE_PEDS.items()
    }


# ---------------------------------------------------------------------------
# oracle guard: the PED-remap helper really tracks the physical axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axcodes", ALL_AXCODES, ids=lambda a: "".join(a))
def test_ped_remap_tracks_physical_axis(axcodes):
    """For every orientation and every physical PE axis, the remapped PED must
    name the voxel axis whose (signed) direction equals the physical axis — the
    invariant the equivariance sweep depends on."""
    affine = _reorient(
        nib.Nifti1Image(np.zeros(_SHAPE, np.float32), _AFFINE), axcodes
    ).affine
    ornt = io_orientation(affine)  # voxel axis -> (world axis, sign)
    for base_world_axis in (0, 1, 2):
        ped = _ped_for_axcodes(axcodes, base_world_axis)
        voxel_axis = AXIS_MAP[ped]
        world_axis, sign = ornt[voxel_axis]
        assert int(world_axis) == base_world_axis
        # a trailing "-" is present iff increasing the voxel index runs against
        # the positive world direction.
        expected_negative = sign < 0
        assert ped.endswith("-") == expected_negative


# ---------------------------------------------------------------------------
# core: orientation equivariance of the full correction over all 48 orientations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axcodes", ALL_AXCODES, ids=lambda a: "".join(a))
def test_correction_is_orientation_equivariant(axcodes, ras_references):
    """Reorient the acquisition into ``axcodes``, remap the PED, run the full
    correction, then reorient the corrected image back to RAS: it must match the
    RAS reference for every base PE axis, i.e. the physical correction does not
    depend on how the volume is stored."""
    fmap_ras, moving_ras = _ras_inputs()
    fmap = _reorient(fmap_ras, axcodes)
    moving = _reorient(moving_ras, axcodes)

    for base_world_axis, reference in ras_references.items():
        ped = _ped_for_axcodes(axcodes, base_world_axis)
        corrected = _corrected(fmap, moving, ped)
        # back to RAS for comparison (scalar image -> lossless reorient)
        back = _reorient(corrected, ("R", "A", "S")).get_fdata()
        assert_allclose(
            back,
            reference,
            atol=1e-4,
            err_msg=(
                f"orientation {''.join(axcodes)} PED {ped!r} (physical axis "
                f"{base_world_axis}) diverged from the RAS correction"
            ),
        )


# ---------------------------------------------------------------------------
# absolute anchors in RAS (so the sweep cannot pass by being uniformly wrong)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pe_dir, channel, lps_sign",
    [
        ("i", 0, -1.0),  # world x: RAS->LPS flip
        ("j", 1, -1.0),  # world y: RAS->LPS flip
        ("k", 2, +1.0),  # world z: no flip
    ],
)
def test_displacement_field_known_value_ras(pe_dir, channel, lps_sign):
    """A constant field map yields a constant displacement field: magnitude
    ``fmap * trt * voxel`` in the PE world channel (carrying the RAS->LPS flip),
    and exactly zero in the other two channels."""
    voxel = (2.0, 3.0, 4.0)
    affine = np.diag([*voxel, 1.0])
    fmap_hz, trt = 10.0, 0.05
    fmap = nib.Nifti1Image(np.full((5, 5, 5), fmap_hz, np.float32), affine)
    dmap = field_maps_to_displacement_maps(fmap, trt, pe_dir)
    field = displacement_map_to_field(dmap, axis=pe_dir, format="itk", frame=0)
    data = field.get_fdata()
    expected = fmap_hz * trt * voxel[channel] * lps_sign
    assert_allclose(data[..., channel], expected, atol=1e-5)
    for other in {0, 1, 2} - {channel}:
        assert_allclose(data[..., other], 0.0, atol=1e-6)


@pytest.mark.parametrize("pe_dir, expected_shift", [("j", -2.0), ("j-", +2.0)])
def test_resample_integer_voxel_shift_ras(pe_dir, expected_shift):
    """With a constant field map chosen to displace by exactly two voxels, a
    linear ramp resamples to that ramp shifted by exactly +-2 voxels along the
    PE axis (linear interpolation is exact on a ramp), and only along that axis.
    Pins the absolute correction direction/magnitude, and that ``j``/``j-`` are
    mirror images."""
    voxel_y = 3.0
    affine = np.diag([2.0, voxel_y, 4.0, 1.0])
    shape = (13, 15, 17)
    # fmap * trt * voxel_y = 20 * 0.1 * 3 = 6 mm = 2 voxels along y
    fmap = nib.Nifti1Image(np.full(shape, 20.0, np.float32), affine)
    dmap = field_maps_to_displacement_maps(fmap, 0.1, pe_dir)
    field = displacement_map_to_field(dmap, axis=pe_dir, format="itk", frame=0)
    ramp = np.broadcast_to(
        np.arange(shape[1], dtype=np.float32)[None, :, None], shape
    ).copy()
    moving = nib.Nifti1Image(ramp, affine)
    out = resample_image(moving, moving, field).get_fdata()

    # interior column along y (avoid the clamped edges), fixed x, z
    interior = slice(4, shape[1] - 4)
    col_in = ramp[6, interior, 8]
    col_out = out[6, interior, 8]
    assert_allclose(col_out - col_in, expected_shift, atol=1e-4)
    # there really is no x/z displacement — assert it on the field directly,
    # since the ramp is flat in x/z and can't reveal off-axis motion via resample.
    field_data = field.get_fdata()
    assert_allclose(field_data[..., AXIS_MAP["x"]], 0.0, atol=1e-6)
    assert_allclose(field_data[..., AXIS_MAP["z"]], 0.0, atol=1e-6)
