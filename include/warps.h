#ifndef WARPS_H
#define WARPS_H

#include <itkComposeImageFilter.h>
#include <itkConstantBoundaryCondition.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkExtractImageFilter.h>
#include <itkHausdorffDistanceImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkModifiedInvertDisplacementFieldImageFilter.h>
#include <itkNthElementImageAdaptor.h>
#include <itkWarpImageFilter.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <utilities.h>

#include <iostream>

namespace py = pybind11;

/**
 * @brief Invert a displacement map
 *
 * @tparam T
 * @param displacement_map
 * @param origin
 * @param direction
 * @param spacing
 * @param axis
 * @param iterations
 * @param verbose
 * @return py::array_t<T, py::array::f_style>
 */
template <typename T>
py::array_t<T, py::array::f_style> invert_displacement_map(py::array_t<T, py::array::f_style> displacement_map,
                                                           py::array_t<T, py::array::f_style> origin,
                                                           py::array_t<T, py::array::f_style> direction,
                                                           py::array_t<T, py::array::f_style> spacing, ssize_t axis,
                                                           ssize_t iterations, bool verbose) {
    // Get axis of map
    ssize_t void_map_axis_0, void_map_axis_1;
    switch (axis) {
        case 0:
            void_map_axis_0 = 1;
            void_map_axis_1 = 2;
            break;
        case 1:
            void_map_axis_0 = 0;
            void_map_axis_1 = 2;
            break;
        case 2:
            void_map_axis_0 = 0;
            void_map_axis_1 = 1;
            break;
        default:
            throw std::invalid_argument("axis must be one of 'x', 'y', or 'z'");
    }

    // Get the displacement map shape
    const ssize_t* shape = displacement_map.shape();

    // Get the displacement map via ImportImageFilter
    using ImportImageFilterType = typename itk::ImportImageFilter<T, 3>;
    using DisplacementMapType = typename ImportImageFilterType::OutputImageType;

    // Setup the displacement map region
    typename DisplacementMapType::IndexType map_index({0, 0, 0});
    using size_value_type = typename DisplacementMapType::SizeValueType;
    typename DisplacementMapType::SizeType map_size({static_cast<size_value_type>(shape[0]),
                                                     static_cast<size_value_type>(shape[1]),
                                                     static_cast<size_value_type>(shape[2])});
    typename DisplacementMapType::RegionType map_region(map_index, map_size);

    // Setup the coordinate system
    typename DisplacementMapType::PointType map_origin({origin.at(0), origin.at(1), origin.at(2)});
    typename DisplacementMapType::DirectionType map_direction;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            map_direction[i][j] = direction.at(i, j);
        }
    }
    typename DisplacementMapType::SpacingType map_spacing({spacing.at(0), spacing.at(1), spacing.at(2)});

    // Setup the ImportImageFilter
    typename ImportImageFilterType::Pointer import_filter = ImportImageFilterType::New();
    import_filter->SetRegion(map_region);
    import_filter->SetOrigin(map_origin);
    import_filter->SetDirection(map_direction);
    import_filter->SetSpacing(map_spacing);
    import_filter->SetImportPointer(displacement_map.mutable_data(), displacement_map.size(), false);

    // create empty displacement map
    typename DisplacementMapType::Pointer void_map = DisplacementMapType::New();
    void_map->SetRegions(map_region);
    void_map->SetOrigin(map_origin);
    void_map->SetDirection(map_direction);
    void_map->SetSpacing(map_spacing);
    void_map->Allocate();
    void_map->FillBuffer(0);

    // Create a Compose Image Filter to create a Displacement Field
    using ComposeImageFilterType =
        typename itk::ComposeImageFilter<DisplacementMapType, itk::Image<itk::Vector<T, 3>, 3>>;
    using DisplacementFieldType = typename ComposeImageFilterType::OutputImageType;

    // create a compose image filter
    typename ComposeImageFilterType::Pointer compose_filter = ComposeImageFilterType::New();
    compose_filter->SetInput(void_map_axis_0, void_map);
    compose_filter->SetInput(void_map_axis_1, void_map);
    compose_filter->SetInput(axis, import_filter->GetOutput());

    // Create an inverse displacement field filter
    using InvertDisplacementFieldFilterType =
        typename itk::ModifiedInvertDisplacementFieldImageFilter<DisplacementFieldType>;
    typename InvertDisplacementFieldFilterType::Pointer invert_displacement_filter =
        InvertDisplacementFieldFilterType::New();

    // Setup invert displacement field filter
    invert_displacement_filter->SetInput(compose_filter->GetOutput());
    invert_displacement_filter->SetMaximumNumberOfIterations(iterations);
    invert_displacement_filter->SetMeanErrorToleranceThreshold(1e-3);
    invert_displacement_filter->SetMaxErrorToleranceThreshold(1e-1);
    invert_displacement_filter->SetEnforceBoundaryCondition(true);
    invert_displacement_filter->SetVerbose(verbose);

    // Create image adaptor to convert field back to map
    using NthElementImageAdaptor = typename itk::NthElementImageAdaptor<DisplacementFieldType, T>;
    typename NthElementImageAdaptor::Pointer nth_element_adaptor = NthElementImageAdaptor::New();
    nth_element_adaptor->SetImage(invert_displacement_filter->GetOutput());
    nth_element_adaptor->SelectNthElement(axis);

    // Make an Identity filter so we can get the output from the adaptor
    using IdentityFilterType = typename itk::ExtractImageFilter<NthElementImageAdaptor, DisplacementMapType>;
    typename IdentityFilterType::Pointer identity_filter = IdentityFilterType::New();
    identity_filter->SetInput(nth_element_adaptor);
    identity_filter->SetExtractionRegion(map_region);
    identity_filter->SetDirectionCollapseToIdentity();

    // Get output
    typename DisplacementMapType::Pointer inv_map = identity_filter->GetOutput();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    inv_map->Update();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    itk::ImageRegionConstIteratorWithIndex<DisplacementMapType> inv_map_it(inv_map,
                                                                           inv_map->GetLargestPossibleRegion());
    py::array_t<T, py::array::f_style> inverted_displacement_map(displacement_map);
    for (inv_map_it.GoToBegin(); !inv_map_it.IsAtEnd(); ++inv_map_it) {
        inverted_displacement_map.mutable_at(inv_map_it.GetIndex()[0], inv_map_it.GetIndex()[1],
                                             inv_map_it.GetIndex()[2]) = inv_map_it.Get();
    }
    return inverted_displacement_map;
}

/**
 * @brief Compute the inverse of a displacement field
 *
 * @tparam T
 * @param displacement_field
 * @param origin
 * @param direction
 * @param spacing
 * @param iterations
 * @param verbose
 * @return py::array_t<T, py::array::f_style>
 */
template <typename T>
py::array_t<T, py::array::f_style> invert_displacement_field(py::array_t<T, py::array::f_style> displacement_field,
                                                             py::array_t<T, py::array::f_style> origin,
                                                             py::array_t<T, py::array::f_style> direction,
                                                             py::array_t<T, py::array::f_style> spacing,
                                                             ssize_t iterations, bool verbose) {
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();

    // Get the displacement field shape
    const ssize_t* shape = displacement_field.shape();

    // Get the displacement field type
    using DisplacementFieldType = typename itk::Image<itk::Vector<T, 3>, 3>;

    // Setup the displacement field region
    typename DisplacementFieldType::IndexType field_index({0, 0, 0});
    using size_value_type = typename DisplacementFieldType::SizeValueType;
    typename DisplacementFieldType::SizeType field_size({static_cast<size_value_type>(shape[0]),
                                                         static_cast<size_value_type>(shape[1]),
                                                         static_cast<size_value_type>(shape[2])});
    typename DisplacementFieldType::RegionType field_region(field_index, field_size);

    // Setup the coordinate system
    typename DisplacementFieldType::PointType field_origin({origin.at(0), origin.at(1), origin.at(2)});
    typename DisplacementFieldType::DirectionType field_direction;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            field_direction[i][j] = direction.at(i, j);
        }
    }
    typename DisplacementFieldType::SpacingType field_spacing({spacing.at(0), spacing.at(1), spacing.at(2)});

    // Create the displacement field and fill it with data
    typename DisplacementFieldType::Pointer field = DisplacementFieldType::New();
    field->SetRegions(field_region);
    field->SetOrigin(field_origin);
    field->SetDirection(field_direction);
    field->SetSpacing(field_spacing);
    field->Allocate();
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> field_it(field, field->GetLargestPossibleRegion());
    for (field_it.GoToBegin(); !field_it.IsAtEnd(); ++field_it) {
        field_it.Set(itk::Vector<T, 3>(
            {displacement_field.at(field_it.GetIndex()[0], field_it.GetIndex()[1], field_it.GetIndex()[2], 0),
             displacement_field.at(field_it.GetIndex()[0], field_it.GetIndex()[1], field_it.GetIndex()[2], 1),
             displacement_field.at(field_it.GetIndex()[0], field_it.GetIndex()[1], field_it.GetIndex()[2], 2)}));
    }

    // Create an inverse displacement field filter
    using InvertDisplacementFieldFilterType =
        typename itk::ModifiedInvertDisplacementFieldImageFilter<DisplacementFieldType>;
    typename InvertDisplacementFieldFilterType::Pointer invert_displacement_filter =
        InvertDisplacementFieldFilterType::New();

    // Setup invert displacement field filter
    invert_displacement_filter->SetInput(field);
    invert_displacement_filter->SetMaximumNumberOfIterations(iterations);
    invert_displacement_filter->SetMeanErrorToleranceThreshold(1e-3);
    invert_displacement_filter->SetMaxErrorToleranceThreshold(1e-1);
    invert_displacement_filter->SetEnforceBoundaryCondition(true);
    invert_displacement_filter->SetVerbose(verbose);

    // Get output
    typename DisplacementFieldType::Pointer inv_field = invert_displacement_filter->GetOutput();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    inv_field->Update();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    itk::ImageRegionConstIteratorWithIndex<DisplacementFieldType> inv_field_it(inv_field,
                                                                               inv_field->GetLargestPossibleRegion());
    py::array_t<T, py::array::f_style> inverted_displacement_field(displacement_field);
    for (inv_field_it.GoToBegin(); !inv_field_it.IsAtEnd(); ++inv_field_it) {
        inverted_displacement_field.mutable_at(inv_field_it.GetIndex()[0], inv_field_it.GetIndex()[1],
                                               inv_field_it.GetIndex()[2], 0) = inv_field_it.Get()[0];
        inverted_displacement_field.mutable_at(inv_field_it.GetIndex()[0], inv_field_it.GetIndex()[1],
                                               inv_field_it.GetIndex()[2], 1) = inv_field_it.Get()[1];
        inverted_displacement_field.mutable_at(inv_field_it.GetIndex()[0], inv_field_it.GetIndex()[1],
                                               inv_field_it.GetIndex()[2], 2) = inv_field_it.Get()[2];
    }

    // Return the inverted displacement field
    return inverted_displacement_field;
}

/**
 * @brief Compute the Jacobian determinant of a displacement field
 *
 * @tparam T
 * @param displacement_field
 * @param origin
 * @param direction
 * @param spacing
 * @return py::array_t<T, py::array::f_style>
 */
template <typename T>
py::array_t<T, py::array::f_style> compute_jacobian_determinant(py::array_t<T, py::array::f_style> displacement_field,
                                                                py::array_t<T, py::array::f_style> origin,
                                                                py::array_t<T, py::array::f_style> direction,
                                                                py::array_t<T, py::array::f_style> spacing) {
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();

    // Get the displacement field shape
    const ssize_t* shape = displacement_field.shape();

    // Get the displacement field type
    using DisplacementFieldType = typename itk::Image<itk::Vector<T, 3>, 3>;

    // Setup the displacement field region
    typename DisplacementFieldType::IndexType field_index({0, 0, 0});
    using size_value_type = typename DisplacementFieldType::SizeValueType;
    typename DisplacementFieldType::SizeType field_size({static_cast<size_value_type>(shape[0]),
                                                         static_cast<size_value_type>(shape[1]),
                                                         static_cast<size_value_type>(shape[2])});
    typename DisplacementFieldType::RegionType field_region(field_index, field_size);

    // Setup the coordinate system
    typename DisplacementFieldType::PointType field_origin({origin.at(0), origin.at(1), origin.at(2)});
    typename DisplacementFieldType::DirectionType field_direction;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            field_direction[i][j] = direction.at(i, j);
        }
    }
    typename DisplacementFieldType::SpacingType field_spacing({spacing.at(0), spacing.at(1), spacing.at(2)});

    // Create the displacement field and fill it with data
    typename DisplacementFieldType::Pointer field = DisplacementFieldType::New();
    field->SetRegions(field_region);
    field->SetOrigin(field_origin);
    field->SetDirection(field_direction);
    field->SetSpacing(field_spacing);
    field->Allocate();
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> field_it(field, field->GetLargestPossibleRegion());
    for (field_it.GoToBegin(); !field_it.IsAtEnd(); ++field_it) {
        field_it.Set(itk::Vector<T, 3>(
            {displacement_field.at(field_it.GetIndex()[0], field_it.GetIndex()[1], field_it.GetIndex()[2], 0),
             displacement_field.at(field_it.GetIndex()[0], field_it.GetIndex()[1], field_it.GetIndex()[2], 1),
             displacement_field.at(field_it.GetIndex()[0], field_it.GetIndex()[1], field_it.GetIndex()[2], 2)}));
    }

    // Create filter for computing jacobian determinant
    using DisplacementFieldJacobianDeterminantFilterType =
        typename itk::DisplacementFieldJacobianDeterminantFilter<DisplacementFieldType, T>;
    typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobian_filter =
        DisplacementFieldJacobianDeterminantFilterType::New();

    // Pass displacment fields into jacobian filters
    jacobian_filter->SetInput(field);
    jacobian_filter->SetUseImageSpacingOff();

    // Get the jacobian determinant fields
    typename DisplacementFieldJacobianDeterminantFilterType::OutputImagePointer jacobian_determinant =
        jacobian_filter->GetOutput();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    jacobian_determinant->Update();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();

    // Convert to numpy array
    py::array_t<T, py::array::f_style> jacobian_determinant_array({shape[0], shape[1], shape[2]});
    itk::ImageRegionConstIteratorWithIndex<typename DisplacementFieldJacobianDeterminantFilterType::OutputImageType>
        jacobian_determinant_it(jacobian_determinant, jacobian_determinant->GetLargestPossibleRegion());
    for (jacobian_determinant_it.GoToBegin(); !jacobian_determinant_it.IsAtEnd(); ++jacobian_determinant_it) {
        jacobian_determinant_array.mutable_at(jacobian_determinant_it.GetIndex()[0],
                                              jacobian_determinant_it.GetIndex()[1],
                                              jacobian_determinant_it.GetIndex()[2]) = jacobian_determinant_it.Get();
    }

    // return jacobian determinant
    return jacobian_determinant_array;
}

/**
 * @brief Resample an image with a given transformation (displacement field)
 *
 * @tparam T
 * @param input_image
 * @param input_origin
 * @param input_direction
 * @param input_spacing
 * @param output_shape
 * @param output_origin
 * @param output_direction
 * @param output_spacing
 * @param transform
 * @param transform_origin
 * @param transform_direction
 * @param transform_spacing
 * @return py::array_t<T, py::array::f_style>
 */
template <typename T>
py::array_t<T, py::array::f_style> resample(
    py::array_t<T, py::array::f_style> input_image, py::array_t<T, py::array::f_style> input_origin,
    py::array_t<T, py::array::f_style> input_direction, py::array_t<T, py::array::f_style> input_spacing,
    py::array_t<T, py::array::f_style> output_shape, py::array_t<T, py::array::f_style> output_origin,
    py::array_t<T, py::array::f_style> output_direction, py::array_t<T, py::array::f_style> output_spacing,
    py::array_t<T, py::array::f_style> transform, py::array_t<T, py::array::f_style> transform_origin,
    py::array_t<T, py::array::f_style> transform_direction, py::array_t<T, py::array::f_style> transform_spacing) {
    // Setup types
    using InputImageType = typename itk::Image<T, 3>;
    using OutputImageType = InputImageType;
    using TransformImageType = typename itk::Image<itk::Vector<T, 3>, 3>;

    // Use ImportImageFilter to import the input image and output image
    using ImportImageFilterType = typename itk::ImportImageFilter<T, 3>;
    typename ImportImageFilterType::Pointer input_import_filter = ImportImageFilterType::New();

    // Setup input import filter
    typename InputImageType::IndexType input_start({0, 0, 0});
    using size_value_type = typename InputImageType::SizeValueType;
    typename InputImageType::SizeType input_size({static_cast<size_value_type>(input_image.shape(0)),
                                                  static_cast<size_value_type>(input_image.shape(1)),
                                                  static_cast<size_value_type>(input_image.shape(2))});
    typename InputImageType::RegionType input_region(input_start, input_size);
    typename InputImageType::PointType input_origin_point({input_origin.at(0), input_origin.at(1), input_origin.at(2)});
    typename InputImageType::DirectionType input_direction_type;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            input_direction_type[i][j] = input_direction.at(i, j);
        }
    }
    typename InputImageType::SpacingType input_spacing_type(
        {input_spacing.at(0), input_spacing.at(1), input_spacing.at(2)});
    input_import_filter->SetRegion(input_region);
    input_import_filter->SetOrigin(input_origin_point);
    input_import_filter->SetDirection(input_direction_type);
    input_import_filter->SetSpacing(input_spacing_type);
    input_import_filter->SetImportPointer(input_image.mutable_data(), input_image.size(), false);

    // Create the transform type and fill with transform input
    typename TransformImageType::Pointer transform_image = TransformImageType::New();
    typename TransformImageType::IndexType transform_start({0, 0, 0});
    using transform_size_value_type = typename TransformImageType::SizeValueType;
    typename TransformImageType::SizeType transform_size({static_cast<transform_size_value_type>(transform.shape(0)),
                                                          static_cast<transform_size_value_type>(transform.shape(1)),
                                                          static_cast<transform_size_value_type>(transform.shape(2))});
    typename TransformImageType::RegionType transform_region(transform_start, transform_size);
    typename TransformImageType::PointType transform_origin_point(
        {transform_origin.at(0), transform_origin.at(1), transform_origin.at(2)});
    typename TransformImageType::DirectionType transform_direction_type;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            transform_direction_type[i][j] = transform_direction.at(i, j);
        }
    }
    typename TransformImageType::SpacingType transform_spacing_type(
        {transform_spacing.at(0), transform_spacing.at(1), transform_spacing.at(2)});
    transform_image->SetRegions(transform_region);
    transform_image->SetOrigin(transform_origin_point);
    transform_image->SetDirection(transform_direction_type);
    transform_image->SetSpacing(transform_spacing_type);
    transform_image->Allocate();
    itk::ImageRegionIteratorWithIndex<TransformImageType> transform_iterator(transform_image, transform_region);
    for (transform_iterator.GoToBegin(); !transform_iterator.IsAtEnd(); ++transform_iterator) {
        transform_iterator.Set(
            itk::Vector<T, 3>({transform.at(transform_iterator.GetIndex()[0], transform_iterator.GetIndex()[1],
                                            transform_iterator.GetIndex()[2], 0),
                               transform.at(transform_iterator.GetIndex()[0], transform_iterator.GetIndex()[1],
                                            transform_iterator.GetIndex()[2], 1),
                               transform.at(transform_iterator.GetIndex()[0], transform_iterator.GetIndex()[1],
                                            transform_iterator.GetIndex()[2], 2)}));
    }

    // Create the WarpImageFilter
    using WarpImageFilterType = typename itk::WarpImageFilter<InputImageType, OutputImageType, TransformImageType>;
    typename WarpImageFilterType::Pointer warp_filter = WarpImageFilterType::New();
    using output_size_value_type = typename OutputImageType::SizeValueType;
    typename OutputImageType::SizeType output_size({static_cast<output_size_value_type>(output_shape.at(0)),
                                                    static_cast<output_size_value_type>(output_shape.at(1)),
                                                    static_cast<output_size_value_type>(output_shape.at(2))});
    typename OutputImageType::PointType output_origin_point(
        {output_origin.at(0), output_origin.at(1), output_origin.at(2)});
    typename OutputImageType::DirectionType output_direction_type;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            output_direction_type[i][j] = output_direction.at(i, j);
        }
    }
    typename OutputImageType::SpacingType output_spacing_type(
        {output_spacing.at(0), output_spacing.at(1), output_spacing.at(2)});
    warp_filter->SetInput(input_import_filter->GetOutput());
    warp_filter->SetOutputSize(output_size);
    warp_filter->SetOutputOrigin(output_origin_point);
    warp_filter->SetOutputDirection(output_direction_type);
    warp_filter->SetOutputSpacing(output_spacing_type);
    warp_filter->SetDisplacementField(transform_image);

    // Create a sinc interpolator
    using BoundaryConditionType = typename itk::ConstantBoundaryCondition<InputImageType>;
    using LanczosWindowType = typename itk::Function::LanczosWindowFunction<5, T, T>;
    using WindowedSincInterpolatorType =
        typename itk::WindowedSincInterpolateImageFunction<InputImageType, 5, LanczosWindowType, BoundaryConditionType,
                                                           T>;

    // Assign interpolator to warp_filter
    typename WindowedSincInterpolatorType::Pointer sinc_interpolator = WindowedSincInterpolatorType::New();
    warp_filter->SetInterpolator(sinc_interpolator);

    // Get the output
    typename OutputImageType::Pointer output_image = warp_filter->GetOutput();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    output_image->Update();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    itk::ImageRegionConstIteratorWithIndex<OutputImageType> output_iterator(output_image,
                                                                            output_image->GetLargestPossibleRegion());
    py::array_t<T, py::array::f_style> output_array({output_shape.at(0), output_shape.at(1), output_shape.at(2)});
    for (output_iterator.GoToBegin(); !output_iterator.IsAtEnd(); ++output_iterator) {
        output_array.mutable_at(output_iterator.GetIndex()[0], output_iterator.GetIndex()[1],
                                output_iterator.GetIndex()[2]) = output_iterator.Get();
    }
    return output_array;
}

/* Compute the Hausdorff Distance between two images */
template <typename T>
T compute_hausdorff_distance(
    py::array_t<T, py::array::f_style> image1, py::array_t<T, py::array::f_style> image1_origin,
    py::array_t<T, py::array::f_style> image1_direction, py::array_t<T, py::array::f_style> image1_spacing,
    py::array_t<T, py::array::f_style> image2, py::array_t<T, py::array::f_style> image2_origin,
    py::array_t<T, py::array::f_style> image2_direction, py::array_t<T, py::array::f_style> image2_spacing) {
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();

    // Setup types
    using ImageType = typename itk::Image<T, 3>;

    // Use ImportImageFilter to import the images
    using ImportImageFilterType = typename itk::ImportImageFilter<T, 3>;
    typename ImportImageFilterType::Pointer input_import_filter1 = ImportImageFilterType::New();
    typename ImportImageFilterType::Pointer input_import_filter2 = ImportImageFilterType::New();

    // Setup input import filter for image 1
    typename ImageType::IndexType input_start1({0, 0, 0});
    using size_value_type = typename ImageType::SizeValueType;
    typename ImageType::SizeType input_size1({static_cast<size_value_type>(image1.shape(0)),
                                              static_cast<size_value_type>(image1.shape(1)),
                                              static_cast<size_value_type>(image1.shape(2))});
    typename ImageType::RegionType image1_region(input_start1, input_size1);
    typename ImageType::PointType image1_origin_point({image1_origin.at(0), image1_origin.at(1), image1_origin.at(2)});
    typename ImageType::DirectionType image1_direction_type;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            image1_direction_type[i][j] = image1_direction.at(i, j);
        }
    }
    typename ImageType::SpacingType image1_spacing_type(
        {image1_spacing.at(0), image1_spacing.at(1), image1_spacing.at(2)});
    input_import_filter1->SetRegion(image1_region);
    input_import_filter1->SetOrigin(image1_origin_point);
    input_import_filter1->SetDirection(image1_direction_type);
    input_import_filter1->SetSpacing(image1_spacing_type);
    input_import_filter1->SetImportPointer(image1.mutable_data(), image1.size(), false);

    // Setup input import filter for image 2
    typename ImageType::IndexType input_start2({0, 0, 0});
    using size_value_type = typename ImageType::SizeValueType;
    typename ImageType::SizeType input_size2({static_cast<size_value_type>(image2.shape(0)),
                                              static_cast<size_value_type>(image2.shape(1)),
                                              static_cast<size_value_type>(image2.shape(2))});
    typename ImageType::RegionType image2_region(input_start2, input_size2);
    typename ImageType::PointType image2_origin_point({image2_origin.at(0), image2_origin.at(1), image2_origin.at(2)});
    typename ImageType::DirectionType image2_direction_type;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            image2_direction_type[i][j] = image2_direction.at(i, j);
        }
    }
    typename ImageType::SpacingType image2_spacing_type(
        {image2_spacing.at(0), image2_spacing.at(1), image2_spacing.at(2)});
    input_import_filter2->SetRegion(image2_region);
    input_import_filter2->SetOrigin(image2_origin_point);
    input_import_filter2->SetDirection(image2_direction_type);
    input_import_filter2->SetSpacing(image2_spacing_type);
    input_import_filter2->SetImportPointer(image2.mutable_data(), image2.size(), false);

    // Create the Hausdorff Distance filter
    using HausdorffDistanceFilterType = typename itk::HausdorffDistanceImageFilter<ImageType, ImageType>;
    typename HausdorffDistanceFilterType::Pointer hausdorff_filter = HausdorffDistanceFilterType::New();
    hausdorff_filter->SetInput1(input_import_filter1->GetOutput());
    hausdorff_filter->SetInput2(input_import_filter2->GetOutput());
    hausdorff_filter->SetUseImageSpacing(true);

    // Get the hausdorff distance
    hausdorff_filter->Update();
    T hausdorff_distance = hausdorff_filter->GetAverageHausdorffDistance();
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();

    // Return the hausdorff distance
    return hausdorff_distance;
}

#endif
