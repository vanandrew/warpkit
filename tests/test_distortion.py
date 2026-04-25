import nibabel as nib
import numpy as np
import pytest
from warpkit.distortion import medic


def test_medic(test_data):
    # this just tests for if the overall medic pipeline errors out
    # should add some assertion checks in the future
    _, _, fmap = medic(
        **test_data,
        n_cpus=1,
    )
    # compute correlations between first frame and all other frames
    # should be a high correlation > 0.98
    data = fmap.get_fdata()
    frame1 = data[..., 0]
    for i in range(1, data.shape[-1]):
        framei = data[..., i]
        corr = np.corrcoef(frame1.ravel(), framei.ravel())[0, 1]
        print(corr)
        assert corr > 0.98, (
            f"Correlation between frame 0 and frame {i} is only {corr} < 0.98"
        )


# ---------------------------------------------------------------------------
# Validation paths — exercised on minimal inputs that fail before the heavy
# unwrap/inversion stages run.
# ---------------------------------------------------------------------------


def _img(affine, shape=(4, 4, 4, 2)):
    return nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)


def test_medic_rejects_mismatched_affines():
    affine_a = np.diag([2.0, 2.0, 2.0, 1.0])
    affine_b = np.diag([3.0, 2.0, 2.0, 1.0])
    phase = [_img(affine_a), _img(affine_b)]
    mag = [_img(affine_a), _img(affine_a)]
    with pytest.raises(ValueError, match="Affines and shapes must match"):
        medic(phase, mag, [14.0, 38.0], 0.05, "j")


def test_medic_rejects_mismatched_shapes():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    phase = [_img(affine, (4, 4, 4, 2)), _img(affine, (4, 4, 5, 2))]
    mag = [_img(affine, (4, 4, 4, 2)), _img(affine, (4, 4, 4, 2))]
    with pytest.raises(ValueError, match="Affines and shapes must match"):
        medic(phase, mag, [14.0, 38.0], 0.05, "j")


def test_medic_rejects_none_affine():
    """A Nifti1Image with no affine is an upstream sentinel for missing
    spatial information; medic must surface that rather than crashing later
    inside np.allclose."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img_a = _img(affine)
    img_b = _img(affine)
    # nibabel exposes affine as a property; we patch the underlying _affine
    # attribute that backs it. type:ignore because we're poking past the API.
    img_b._affine = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="non-None affine"):
        medic([img_a, img_b], [img_a, img_b], [14.0, 38.0], 0.05, "j")
