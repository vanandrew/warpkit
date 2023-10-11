import argparse
import nibabel as nib
from warpkit.utilities import AXIS_MAP, displacement_map_to_field, WARP_ITK_FLIPS
from . import epilog


def main():
    parser = argparse.ArgumentParser(
        description="This program extracts a displacement field from a series of displacement maps.",
        epilog=f"{epilog} 12/14/2022",
    )
    parser.add_argument("maps", help="Displacement maps to extract field from.")
    parser.add_argument("field", help="Displacement field to write out.")
    parser.add_argument(
        "-n",
        "--frame_number",
        type=int,
        default=0,
        help="Frame number to extract field from. 0-indexed. By default 0th frame.",
    )
    parser.add_argument(
        "-p",
        "--phase_encoding_axis",
        default="j",
        choices=[d for d in AXIS_MAP.keys()],
        help="The phase encoding axis of the data. Default is j.",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="itk",
        choices=[f for f in WARP_ITK_FLIPS.keys()],
    )

    # parse arguments
    args = parser.parse_args()

    # load the displacement maps
    maps_img = nib.load(args.maps)

    # grab the map specified by frame_number
    selected_map_data = maps_img.dataobj[:, :, :, args.frame_number]

    # make a new nifti image with the selected map
    selected_map_img = nib.Nifti1Image(selected_map_data, maps_img.affine, maps_img.header)

    # transform map to field (specified by phase_encoding_axis and file format)
    field_img = displacement_map_to_field(selected_map_img, args.phase_encoding_axis, args.format)

    # save the field
    field_img.to_filename(args.field)
