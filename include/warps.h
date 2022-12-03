#ifndef WARPS_H
#define WARPS_H

#include <itkComposeImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkModifiedInvertDisplacementFieldImageFilter.h>
#include <itkNthElementImageAdaptor.h>
#include <itkWarpImageFilter.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <utilities.h>

#include <iostream>
#include <vector>

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
    std::vector<T> inv_map_data(inv_map->GetBufferPointer(),
                                inv_map->GetBufferPointer() + inv_map->GetBufferedRegion().GetNumberOfPixels());
    return as_pyarray(inv_map_data, {shape[0], shape[1], shape[2]});
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

    // Get the displacement map via ImportImageFilter
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
    py::array_t<T, py::array::f_style> invert_displacement_field(displacement_field);
    for (inv_field_it.GoToBegin(); !inv_field_it.IsAtEnd(); ++inv_field_it) {
        invert_displacement_field.mutable_at(inv_field_it.GetIndex()[0], inv_field_it.GetIndex()[1],
                                             inv_field_it.GetIndex()[2], 0) = inv_field_it.Get()[0];
        invert_displacement_field.mutable_at(inv_field_it.GetIndex()[0], inv_field_it.GetIndex()[1],
                                             inv_field_it.GetIndex()[2], 1) = inv_field_it.Get()[1];
        invert_displacement_field.mutable_at(inv_field_it.GetIndex()[0], inv_field_it.GetIndex()[1],
                                             inv_field_it.GetIndex()[2], 2) = inv_field_it.Get()[2];
    }
    return invert_displacement_field;
}

#endif
