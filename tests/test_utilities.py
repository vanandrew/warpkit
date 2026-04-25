import logging

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from warpkit.utilities import (
    AXIS_MAP,
    WARP_ITK_FLIPS,
    build_low_pass_filter,
    compute_hausdorff_distance,
    compute_jacobian_determinant,
    convert_warp,
    corr2_coeff,
    create_brain_mask,
    displacement_field_to_map,
    displacement_map_to_field,
    displacement_maps_to_field_maps,
    field_maps_to_displacement_maps,
    get_largest_connected_component,
    get_ras_orient_transform,
    get_x_orient_transform,
    invert_displacement_field,
    invert_displacement_maps,
    normalize,
    rescale_phase,
    setup_logging,
)

# ---------------------------------------------------------------------------
# normalize / rescale_phase
# ---------------------------------------------------------------------------


def test_normalize_maps_to_unit_interval():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = normalize(data)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert_allclose(out, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_rescale_phase_default_range():
    # default min=-4096, max=4096 maps midpoint (0) to 0 rad, edges to ±π.
    assert rescale_phase(np.array([0.0])) == pytest.approx(0.0, abs=1e-6)
    assert rescale_phase(np.array([-4096.0])) == pytest.approx(-np.pi)
    assert rescale_phase(np.array([4096.0])) == pytest.approx(np.pi)


# ---------------------------------------------------------------------------
# corr2_coeff
# ---------------------------------------------------------------------------


def test_corr2_coeff_identical():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    b = a.copy()
    assert_allclose(corr2_coeff(a, b), np.ones((3, 3)))


def test_corr2_coeff_negative():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    b = -a
    assert_allclose(corr2_coeff(a, b), -np.ones((3, 3)))


def test_corr2_coeff_known():
    rng = np.random.default_rng(42)
    size = 10
    a = rng.random((size, size))
    b = rng.random((size, 1))
    expected = np.array([np.corrcoef(b[:, 0], a[:, i])[0, 1] for i in range(size)])
    assert_allclose(corr2_coeff(b, a).ravel(), expected)


# ---------------------------------------------------------------------------
# masking helpers
# ---------------------------------------------------------------------------


def test_get_largest_connected_component():
    # two blobs: one big (8 voxels), one small (1 voxel). Largest wins.
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:3, 1:3, 1:3] = True  # 2x2x2 = 8 voxels
    mask[5, 5, 5] = True  # isolated singleton
    out = get_largest_connected_component(mask)
    assert out.sum() == 8
    assert not out[5, 5, 5]
    assert out[1, 1, 1]


def test_create_brain_mask_basic():
    # construct a synthetic gaussian-ish "brain" centered in volume.
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    coords = np.indices(shape).astype(np.float32)
    center = np.array([7.5, 7.5, 7.5])
    r2 = sum((coords[i] - center[i]) ** 2 for i in range(3))
    mag = np.exp(-r2 / 30.0).astype(np.float32) + rng.normal(0, 0.01, shape).astype(
        np.float32
    )
    mask = create_brain_mask(mag)
    # mask must be bool, non-empty, and concentrated around the center.
    assert mask.dtype == np.bool_
    assert mask.any()
    assert mask[7, 7, 7]
    # outermost corner should not be in mask
    assert not mask[0, 0, 0]


# ---------------------------------------------------------------------------
# orientation transforms
# ---------------------------------------------------------------------------


def _ras_image(shape=(4, 4, 4)) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))


def test_get_ras_orient_transform_identity_on_ras_image():
    img = _ras_image()
    to_canonical, from_canonical = get_ras_orient_transform(img)
    # nibabel's ornt_transform returns ndarray at runtime even though our
    # signature now claims Sequence[Sequence[int]].
    to_arr = np.asarray(to_canonical)
    from_arr = np.asarray(from_canonical)
    expected = np.array([[0, 1], [1, 1], [2, 1]])
    assert_allclose(to_arr, expected)
    assert_allclose(from_arr, expected)


def test_get_x_orient_transform_roundtrip_lps():
    # LPS is RAS with x and y axes flipped. Round trip should land back.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), affine)
    to_canonical, from_canonical = get_x_orient_transform(img, "RAS")
    reoriented = img.as_reoriented(to_canonical).as_reoriented(from_canonical)
    # nibabel types affine as Optional[ndarray]; we know both inputs are real.
    assert reoriented.affine is not None and img.affine is not None
    assert_allclose(reoriented.affine, img.affine)
    assert reoriented.shape == img.shape


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_setup_logging_with_file(tmp_path):
    log_file = tmp_path / "subdir" / "warpkit.log"
    setup_logging(str(log_file))
    # creates the parent directory and writes a file handler that points at it.
    assert log_file.parent.exists()
    # log a message and verify it lands.
    logging.getLogger().info("hello-from-test")
    # FileHandler buffers; flush all root handlers.
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_file.exists()


def test_setup_logging_no_file_does_not_raise():
    # smoke test: the no-file branch just adds a stdout handler.
    setup_logging(None)


# ---------------------------------------------------------------------------
# build_low_pass_filter
# ---------------------------------------------------------------------------


def test_build_low_pass_filter_returns_butterworth():
    b, a = build_low_pass_filter(tr_in_sec=1.0, critical_freq=0.1, filter_order=4)
    # Butterworth IIR with order 4 → 5-tap b and a coefficients.
    assert b.shape == (5,)
    assert a.shape == (5,)
    # a[0] is normalized to 1.0 for SciPy butterworth output.
    assert a[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# field map / displacement map round trips
# ---------------------------------------------------------------------------


def _synthetic_4d_image(shape=(4, 4, 4, 2), seed=42, scale=5.0):
    rng = np.random.default_rng(seed)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-3.0, -3.0, -3.0]
    data = (rng.standard_normal(shape) * scale).astype(np.float32)
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("pe_dir", ["i", "j", "k", "i-", "j-", "k-"])
def test_field_displacement_round_trip(pe_dir):
    """fmap → dmap → fmap recovers the original (no flip_sign)."""
    fmap = _synthetic_4d_image()
    total_readout_time = 0.05
    dmap = field_maps_to_displacement_maps(fmap, total_readout_time, pe_dir)
    fmap2 = displacement_maps_to_field_maps(dmap, total_readout_time, pe_dir)
    assert_allclose(fmap2.get_fdata(), fmap.get_fdata(), rtol=1e-4)


def test_displacement_maps_to_field_maps_flip_sign():
    fmap = _synthetic_4d_image()
    dmap = field_maps_to_displacement_maps(fmap, 0.05, "j")
    fmap_flipped = displacement_maps_to_field_maps(dmap, 0.05, "j", flip_sign=True)
    assert_allclose(fmap_flipped.get_fdata(), -fmap.get_fdata(), rtol=1e-4)


def test_displacement_map_to_field_inserts_along_axis():
    """A scalar displacement map at axis="y" must end up entirely in the y component
    of the resulting 3-vector field (modulo the warp-format sign flip applied for
    the default itk output)."""
    rng = np.random.default_rng(0)
    affine = np.eye(4)
    data = rng.standard_normal((4, 4, 4, 2)).astype(np.float32)
    dmap = nib.Nifti1Image(data, affine)
    field = displacement_map_to_field(dmap, axis="y", format="itk", frame=0)
    field_data = field.get_fdata().squeeze()
    # itk format applies WARP_ITK_FLIPS["itk"] = [1, 1, 1] → identity; y component
    # equals the source frame, x and z are zero.
    assert_allclose(field_data[..., AXIS_MAP["y"]], data[..., 0])
    assert_allclose(field_data[..., AXIS_MAP["x"]], 0.0)
    assert_allclose(field_data[..., AXIS_MAP["z"]], 0.0)


# ---------------------------------------------------------------------------
# displacement field / map inversion (exercises the C++ path)
# ---------------------------------------------------------------------------


def _zero_3vec_field(shape=(8, 8, 8)) -> nib.Nifti1Image:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    return nib.Nifti1Image(np.zeros((*shape, 3), dtype=np.float32), affine)


def test_invert_displacement_field_zero():
    """Inverting a zero field gives back zero (within numerical noise) and
    preserves the 3-channel last axis."""
    field = _zero_3vec_field()
    inverted = invert_displacement_field(field)
    assert inverted.shape == field.shape
    assert_allclose(inverted.get_fdata(), 0.0, atol=1e-5)


def test_invert_displacement_maps_zero():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    dmap = nib.Nifti1Image(np.zeros((6, 6, 6, 2), dtype=np.float32), affine)
    inverted = invert_displacement_maps(dmap, axis="y")
    assert inverted.shape == dmap.shape
    assert_allclose(inverted.get_fdata(), 0.0, atol=1e-5)


def test_compute_jacobian_determinant_zero_field():
    """The Jacobian determinant of a zero displacement field is identically 1
    (the identity transform doesn't change volume). Exercises the
    compute_jacobian_determinant_cpp binding."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field = nib.Nifti1Image(np.zeros((6, 6, 6, 3), dtype=np.float32), affine)
    jdet = compute_jacobian_determinant(field)
    assert jdet.shape == (6, 6, 6)
    assert_allclose(jdet.get_fdata(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------


def test_compute_hausdorff_distance_identical_is_zero():
    """Hausdorff(image, image) == 0."""
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.float64), affine)
    d = compute_hausdorff_distance(img, img)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_compute_hausdorff_distance_symmetric():
    """Hausdorff(a, b) should equal Hausdorff(b, a)."""
    rng = np.random.default_rng(0)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    a_data = (rng.random((6, 6, 6)) > 0.5).astype(np.float64)
    b_data = (rng.random((6, 6, 6)) > 0.5).astype(np.float64)
    a = nib.Nifti1Image(a_data, affine)
    b = nib.Nifti1Image(b_data, affine)
    d_ab = compute_hausdorff_distance(a, b)
    d_ba = compute_hausdorff_distance(b, a)
    assert d_ab == pytest.approx(d_ba, abs=1e-6)


# ---------------------------------------------------------------------------
# convert_warp
# ---------------------------------------------------------------------------


def test_convert_warp_itk_to_itk_is_identity():
    """itk → itk applies the same flip array twice = identity (modulo dim shape)."""
    affine = np.eye(4)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 4, 4, 3)).astype(np.float32)
    warp = nib.Nifti1Image(data, affine)
    out = convert_warp(warp, in_type="itk", out_type="itk")
    assert_allclose(out.get_fdata(), data, atol=1e-5)


def test_convert_warp_fsl_to_itk_flips_x():
    """fsl → itk flips x sign (WARP_ITK_FLIPS['fsl'] = [-1, 1, 1])."""
    affine = np.eye(4)
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 4, 4, 3)).astype(np.float32)
    fsl_warp = nib.Nifti1Image(data, affine)
    out = convert_warp(fsl_warp, in_type="fsl", out_type="itk")
    out_data = out.get_fdata()
    flips = WARP_ITK_FLIPS["fsl"]
    expected = data.copy()
    for ax in range(3):
        expected[..., ax] *= flips[ax]
    assert_allclose(out_data, expected, atol=1e-5)


def test_convert_warp_rejects_bad_shape():
    affine = np.eye(4)
    bad = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine)
    with pytest.raises(ValueError, match="must be 4D or 5D"):
        convert_warp(bad, "itk", "itk")


def test_convert_warp_rejects_bad_last_axis():
    affine = np.eye(4)
    bad = nib.Nifti1Image(np.zeros((4, 4, 4, 2), dtype=np.float32), affine)
    with pytest.raises(ValueError, match="size 3 in last axis"):
        convert_warp(bad, "itk", "itk")


def test_convert_warp_rejects_unknown_type():
    affine = np.eye(4)
    warp = nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), affine)
    with pytest.raises(ValueError, match="not recognized"):
        convert_warp(warp, in_type="bogus", out_type="itk")


def test_convert_warp_accepts_5d_ants_layout():
    """ANTs/AFNI single warps are 5D with shape (X, Y, Z, 1, 3); convert_warp
    must accept that layout."""
    affine = np.eye(4)
    rng = np.random.default_rng(0)
    data_5d = rng.standard_normal((4, 4, 4, 1, 3)).astype(np.float32)
    warp = nib.Nifti1Image(data_5d, affine)
    out = convert_warp(warp, in_type="ants", out_type="itk")
    # ants -> itk flips x and y (WARP_ITK_FLIPS["ants"] = [-1, -1, 1])
    expected = data_5d[..., 0, :].copy()
    expected[..., 0] *= -1
    expected[..., 1] *= -1
    assert_allclose(out.get_fdata().squeeze(), expected, atol=1e-5)


def test_convert_warp_rejects_bad_5d_shape():
    """A 5D warp without the (X,Y,Z,1,3) layout (e.g. multi-frame 5D series)
    must be rejected — the per-frame split should have happened upstream."""
    affine = np.eye(4)
    bad = nib.Nifti1Image(np.zeros((4, 4, 4, 2, 3), dtype=np.float32), affine)
    with pytest.raises(ValueError, match="singleton 4th axis"):
        convert_warp(bad, "itk", "itk")


@pytest.mark.parametrize("fmt", ["itk", "fsl", "ants", "afni"])
@pytest.mark.parametrize("axis", ["x", "y", "z", "x-", "y-", "z-"])
def test_displacement_map_field_roundtrip_all_formats(axis, fmt):
    """displacement_map_to_field followed by displacement_field_to_map must
    recover the original 1-channel map, for every (axis, format) pair —
    including the ants/afni formats whose intermediate field is 5D."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((4, 4, 4)).astype(np.float32)
    dmap = nib.Nifti1Image(data, np.eye(4))
    field = displacement_map_to_field(dmap, axis=axis, format=fmt, frame=0)
    back = displacement_field_to_map(field, axis=axis, format=fmt)
    assert_allclose(back.get_fdata(), data, atol=1e-5)


def test_displacement_field_to_map_clears_vector_intent():
    """The 1-channel result must not carry the input field's vector intent,
    or downstream type checks may mistake it for a field."""
    affine = np.eye(4)
    data = np.zeros((4, 4, 4), dtype=np.float32)
    dmap = nib.Nifti1Image(data, affine)
    field = displacement_map_to_field(dmap, axis="y", format="itk", frame=0)
    # The intermediate field carries the vector intent (set by convert_warp).
    assert field.header.get_intent()[0] == "vector"
    back = displacement_field_to_map(field, axis="y", format="itk")
    assert back.header.get_intent()[0] != "vector"


# ---------------------------------------------------------------------------
# Quantitative Hz <-> mm tests with hand-computed expected values.
# Formula: displacement = fieldmap * total_readout_time * voxel_size, where
# voxel_size carries the LPS-x/y sign flip (axes 0/1) and the negative-PE flip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pe_dir, expected_sign",
    [
        ("i", -1.0),  # axis 0: itk LPS x flip
        ("i-", +1.0),  # axis 0: "-" cancels itk flip
        ("j", -1.0),  # axis 1: itk LPS y flip
        ("j-", +1.0),  # axis 1: "-" cancels itk flip
        ("k", +1.0),  # axis 2: no LPS flip
        ("k-", -1.0),  # axis 2: just "-"
    ],
)
def test_field_maps_to_displacement_maps_known_value(pe_dir, expected_sign):
    """For a constant fmap with isotropic voxels, the displacement equals
    fmap * trt * voxel * (sign per PE convention)."""
    voxel = 2.0
    affine = np.diag([voxel, voxel, voxel, 1.0])
    fmap_value = 10.0  # Hz
    trt = 0.05  # s
    fmap_data = np.full((4, 4, 4), fmap_value, dtype=np.float32)
    fmap = nib.Nifti1Image(fmap_data, affine)
    dmap = field_maps_to_displacement_maps(fmap, trt, pe_dir)
    expected = fmap_value * trt * voxel * expected_sign
    assert_allclose(dmap.get_fdata(), expected, atol=1e-5)


@pytest.mark.parametrize("axis", ["i", "j", "k"])
def test_field_maps_to_displacement_maps_anisotropic_voxels(axis):
    """The voxel size used for scaling must come from the PE-axis dimension,
    not a different axis."""
    voxels = (1.0, 2.0, 3.0)
    affine = np.diag([*voxels, 1.0])
    fmap = nib.Nifti1Image(np.full((4, 4, 4), 10.0, dtype=np.float32), affine)
    dmap = field_maps_to_displacement_maps(fmap, 0.05, axis)
    axis_idx = AXIS_MAP[axis]
    expected_mag = 10.0 * 0.05 * voxels[axis_idx]
    assert_allclose(np.abs(dmap.get_fdata()), expected_mag, atol=1e-5)


@pytest.mark.parametrize("axis", ["i", "j", "k"])
def test_field_maps_to_displacement_maps_pe_sign_negation(axis):
    """For the same fmap and voxels, '<axis>' and '<axis>-' produce results
    that are exact negations of each other."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    fmap = nib.Nifti1Image(rng.standard_normal((4, 4, 4)).astype(np.float32), affine)
    pos = field_maps_to_displacement_maps(fmap, 0.05, axis)
    neg = field_maps_to_displacement_maps(fmap, 0.05, f"{axis}-")
    assert_allclose(pos.get_fdata(), -neg.get_fdata(), atol=1e-6)


# ---------------------------------------------------------------------------
# displacement_map_to_field frame selection
# ---------------------------------------------------------------------------


def test_displacement_map_to_field_frame_selects_correct_frame():
    """A 4D displacement map series + frame=N yields a 4D field whose
    PE-axis channel matches frame N of the source."""
    affine = np.eye(4)
    rng = np.random.default_rng(0)
    map_data = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
    dmap = nib.Nifti1Image(map_data, affine)
    for frame in range(map_data.shape[-1]):
        field = displacement_map_to_field(dmap, axis="z", format="itk", frame=frame)
        # itk format with axis=z (no sign flips) → channel 2 == source frame
        assert_allclose(field.get_fdata()[..., 2], map_data[..., frame], atol=1e-6)


# ---------------------------------------------------------------------------
# displacement_field_to_map drops off-axis channels (documented behavior)
# ---------------------------------------------------------------------------


def test_displacement_field_to_map_drops_off_axis_channels():
    """For an EPI distortion warp all displacement is along the PE axis, so
    extracting one channel is exact. Confirm that a field with non-zero values
    on every channel has the off-axis values dropped."""
    affine = np.eye(4)
    rng = np.random.default_rng(0)
    field_data = rng.standard_normal((4, 4, 4, 3)).astype(np.float32)
    field = nib.Nifti1Image(field_data, affine)
    # itk format = identity → extracted map equals the chosen channel verbatim.
    extracted_y = displacement_field_to_map(field, axis="y", format="itk").get_fdata()
    assert_allclose(extracted_y, field_data[..., 1], atol=1e-6)


# ---------------------------------------------------------------------------
# convert_warp roundtrip across non-canonical orientations
# ---------------------------------------------------------------------------


def test_convert_warp_lps_affine_itk_roundtrip():
    """itk -> itk on an image with an LPS affine must preserve data exactly
    (the as_reoriented round trip should land back on the original grid)."""
    # LPS affine: x and y axes flipped vs RAS
    affine = np.diag([-2.0, -2.0, 2.0, 1.0])
    affine[:3, 3] = [10.0, 20.0, -5.0]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 4, 4, 3)).astype(np.float32)
    warp = nib.Nifti1Image(data, affine)
    out = convert_warp(warp, in_type="itk", out_type="itk")
    assert_allclose(out.get_fdata(), data, atol=1e-5)
    # Affine and shape must be preserved on the round trip.
    assert out.affine is not None and warp.affine is not None
    assert_allclose(out.affine, warp.affine)
    assert out.shape == warp.shape


# ---------------------------------------------------------------------------
# compute_jacobian_determinant on a non-zero translation field
# ---------------------------------------------------------------------------


def test_convert_warp_rejects_unknown_output_type():
    """A bogus out_type triggers the post-flip ValueError (separate from the
    in_type unknown branch)."""
    affine = np.eye(4)
    warp = nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), affine)
    with pytest.raises(ValueError, match="not recognized"):
        convert_warp(warp, in_type="itk", out_type="bogus")


def test_resample_image_rejects_transform_with_wrong_last_axis():
    """resample_image checks the transform data has size 3 in the last axis
    after the optional 5D squeeze."""
    from warpkit.utilities import resample_image

    affine = np.eye(4)
    ref = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine)
    inp = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine)
    bad_transform = nib.Nifti1Image(np.zeros((4, 4, 4, 7), dtype=np.float32), affine)
    with pytest.raises(ValueError, match="size 3 in last axis"):
        resample_image(ref, inp, bad_transform)


def test_resample_image_squeezes_5d_transform():
    """A 5D ANTs/AFNI transform (X,Y,Z,1,3) is squeezed to 4D before resample.
    Identity transform → output equals input."""
    from warpkit.utilities import resample_image

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    img_data = rng.random((6, 6, 6), dtype=np.float32)
    inp = nib.Nifti1Image(img_data, affine)
    ref = nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), affine)
    transform = nib.Nifti1Image(np.zeros((6, 6, 6, 1, 3), dtype=np.float32), affine)
    out = resample_image(ref, inp, transform)
    assert_allclose(out.get_fdata().squeeze(), img_data, atol=1e-3)


def test_create_brain_mask_negative_extra_dilation_erodes():
    """A negative extra_dilation triggers the erosion branch in
    create_brain_mask."""
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    coords = np.indices(shape).astype(np.float32)
    center = np.array([7.5, 7.5, 7.5])
    r2 = sum((coords[i] - center[i]) ** 2 for i in range(3))
    mag = np.exp(-r2 / 30.0).astype(np.float32) + rng.normal(0, 0.01, shape).astype(
        np.float32
    )
    mask_default = create_brain_mask(mag)
    mask_eroded = create_brain_mask(mag, extra_dilation=-2)
    # Erosion strictly shrinks the mask.
    assert mask_eroded.sum() < mask_default.sum()


def test_create_brain_mask_extra_dilation_grows_mask():
    """A positive extra_dilation strictly grows the mask."""
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    coords = np.indices(shape).astype(np.float32)
    center = np.array([7.5, 7.5, 7.5])
    r2 = sum((coords[i] - center[i]) ** 2 for i in range(3))
    mag = np.exp(-r2 / 30.0).astype(np.float32) + rng.normal(0, 0.01, shape).astype(
        np.float32
    )
    mask_default = create_brain_mask(mag)
    mask_dilated = create_brain_mask(mag, extra_dilation=3)
    assert mask_dilated.sum() > mask_default.sum()


def test_compute_jacobian_determinant_constant_translation_is_one():
    """A spatially constant displacement is a pure translation and has unit
    Jacobian determinant. Complements the all-zero test by exercising the
    non-zero code path."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field = np.zeros((8, 8, 8, 3), dtype=np.float32)
    field[..., 0] = 3.5  # constant x-shift
    field[..., 1] = -1.0  # constant y-shift
    field[..., 2] = 0.5  # constant z-shift
    img = nib.Nifti1Image(field, affine)
    jdet = compute_jacobian_determinant(img)
    # Trim a 1-voxel border to skip any boundary differencing artifacts.
    interior = jdet.get_fdata()[1:-1, 1:-1, 1:-1]
    assert_allclose(interior, 1.0, atol=1e-4)
