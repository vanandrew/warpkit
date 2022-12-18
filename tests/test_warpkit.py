from warpkit.distortion import me_sdc
from warpkit.utilities import displacement_map_to_field, invert_displacement_field, resample_image  # type: ignore
from . import *


def test_me_sdc(test_data):
    displacement_maps = me_sdc(
        [i.get_image() for i in test_data["phase"]],
        [i.get_image() for i in test_data["mag"]],
        test_data["TEs"],
        test_data["TotalReadoutTime"],
        test_data["PhaseEncodingDirection"],
        up_to_frame=5,
    )
    displacement_maps.to_filename("/home/vanandrew/Data/test.nii.gz")


def test_resample_image(test_data, test_warp):
    import numpy as np
    import nibabel as nib

    data = test_warp.dataobj[..., 0]
    map_img = nib.Nifti1Image(data, test_warp.affine)
    warp_img = displacement_map_to_field(map_img)
    warp_img.to_filename("/home/vanandrew/Data/test_warp.nii.gz")

    mag_img = test_data["mag"][0].get_image()
    in_img = nib.Nifti1Image(mag_img.dataobj[..., 0], mag_img.affine, mag_img.header)
    in_img.to_filename("/home/vanandrew/Data/test_in.nii.gz")
    resample_image(in_img, in_img, warp_img).to_filename("/home/vanandrew/Data/test_resamp.nii.gz")
