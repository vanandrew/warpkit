from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = [
    "Romeo",
    "compute_hausdorff_distance",
    "compute_jacobian_determinant",
    "invert_displacement_field",
    "invert_displacement_map",
    "resample",
]

class Romeo:
    def __init__(self) -> None: ...
    def calculate_weights(
        self,
        phase: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        mag: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] = ...,
        phase2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] = ...,
        tes: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] = ...,
        mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool] = ...,
    ) -> numpy.typing.NDArray[numpy.uint8]:
        """
        ROMEO edge-weight map (3, nx, ny, nz) uint8. Exposed for port validation; not used by warpkit.
        """
    def romeo_unwrap3d(
        self,
        phase: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        weights: str,
        mag: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool],
        correct_global: bool = True,
        maxseeds: typing.SupportsInt | typing.SupportsIndex = 1,
        merge_regions: bool = False,
        correct_regions: bool = False,
    ) -> numpy.typing.NDArray[numpy.float32]:
        """
        3D ROMEO phase unwrap
        """
    def romeo_unwrap4d(
        self,
        phase: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        tes: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        weights: str,
        mag: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool],
        correct_global: bool = True,
        maxseeds: typing.SupportsInt | typing.SupportsIndex = 1,
        merge_regions: bool = False,
        correct_regions: bool = False,
    ) -> numpy.typing.NDArray[numpy.float32]:
        """
        4D (multi-echo) ROMEO phase unwrap
        """
    def romeo_voxelquality(
        self,
        phase: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        tes: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        mag: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    ) -> numpy.typing.NDArray[numpy.float32]:
        """
        Compute a per-voxel quality map from multi-echo phase/magnitude
        """

def compute_hausdorff_distance(
    image1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image1_origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image1_direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image1_spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image2_origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image2_direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    image2_spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> float:
    """
    Compute the Hausdorff Distance
    """

def compute_jacobian_determinant(
    displacement_field: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Compute the Jacobian determinant of a displacement field
    """

def invert_displacement_field(
    displacement_field: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    iterations: typing.SupportsInt | typing.SupportsIndex = 50,
    verbose: bool = False,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Invert a displacement field
    """

def invert_displacement_map(
    displacement_map: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    axis: typing.SupportsInt | typing.SupportsIndex = 1,
    iterations: typing.SupportsInt | typing.SupportsIndex = 50,
    verbose: bool = False,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Invert a displacement map
    """

def resample(
    input_image: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    input_origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    input_direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    input_spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    output_shape: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    output_origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    output_direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    output_spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    transform: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    transform_origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    transform_direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    transform_spacing: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Resample an image with transform
    """
