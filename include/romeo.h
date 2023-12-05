#ifndef ROMEO_H
#define ROMEO_H

#include <julia.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <utilities.h>

#include <csignal>
#include <iostream>
#include <sstream>
#include <string>

namespace py = pybind11;

/**
 * @brief Class to call julia functions from python
 *
 * @tparam T
 */
template <typename T>
class JuliaContext {
   public:
    /**
     * @brief Construct a new Julia Context object
     *
     */
    JuliaContext() {
        // initialize Julia context
        jl_init();

        // setup types
        jl_value_t* ntuple3[3] = {reinterpret_cast<jl_value_t*>(jl_long_type),
                                  reinterpret_cast<jl_value_t*>(jl_long_type),
                                  reinterpret_cast<jl_value_t*>(jl_long_type)};
        jl_ntuple3 = jl_apply_tuple_type_v(ntuple3, 3);
        jl_value_t* ntuple4[4] = {
            reinterpret_cast<jl_value_t*>(jl_long_type), reinterpret_cast<jl_value_t*>(jl_long_type),
            reinterpret_cast<jl_value_t*>(jl_long_type), reinterpret_cast<jl_value_t*>(jl_long_type)};
        jl_ntuple4 = jl_apply_tuple_type_v(ntuple4, 4);
        jl_array3d_bool = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_bool_type), 3);
        if (std::is_same<T, double>::value)
            jl_vector = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 1);
        else if (std::is_same<T, float>::value)
            jl_vector = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float32_type), 1);
        if (std::is_same<T, double>::value)
            jl_array3d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 3);
        else if (std::is_same<T, float>::value)
            jl_array3d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float32_type), 3);
        if (std::is_same<T, double>::value)
            jl_array4d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 4);
        else if (std::is_same<T, float>::value)
            jl_array4d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float32_type), 4);

        // initialize modules
        jl_eval_string("using ROMEO;");

        // get functions from modules
        // wrap them in lambdas so we can call arguments positionally
        jl_voxelquality =
            static_cast<jl_function_t*>(jl_eval_string("voxelquality_positional_wrapper(phase, TEs, mag) = "
                                                       "replace!(Float32.(voxelquality(Float32.(phase); "
                                                       "TEs=Float32.(TEs), mag=Float32.(mag))), NaN=>0);"));
        jl_unwrap3D = static_cast<jl_function_t*>(
            jl_eval_string("unwrap3D_positional_wrapper(phase, weights, mag, mask, correctglobal, "
                           "maxseeds, merge_regions, correct_regions) = unwrap(phase, "
                           "weights=weights, mag=mag, mask=mask, correctglobal=correctglobal, maxseeds=maxseeds, "
                           "merge_regions=merge_regions, correct_regions=correct_regions);"));
        jl_unwrap4D = static_cast<jl_function_t*>(
            jl_eval_string("unwrap4D_positional_wrapper(phase, TEs, weights, mag, mask, correctglobal, "
                           "maxseeds, merge_regions, correct_regions) = unwrap(phase, TEs=TEs, "
                           "weights=weights, mag=mag, mask=mask, correctglobal=correctglobal, maxseeds=maxseeds, "
                           "merge_regions=merge_regions, correct_regions=correct_regions);"));
    }

    /**
     * @brief Wrapper for ROMEO voxel_quality function
     *
     * @param phase
     * @param TEs
     * @param weights
     * @param mag
     * @param mask
     * @return py::array_t<T, py::array::f_style>
     */
    py::array_t<T, py::array::f_style> romeo_voxelquality(py::array_t<T, py::array::f_style> phase,
                                                          py::array_t<T, py::array::f_style> TEs,
                                                          py::array_t<T, py::array::f_style> mag) {
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();

        // Get dimensions as julia tuples
        jl_ntuple4_t* phase_dims = reinterpret_cast<jl_ntuple4_t*>(jl_new_struct_uninit(jl_ntuple4));
        jl_ntuple4_t* mag_dims = reinterpret_cast<jl_ntuple4_t*>(jl_new_struct_uninit(jl_ntuple4));
        jl_array_t* jl_phase;
        jl_array_t* jl_TEs;
        jl_array_t* jl_mag;
        JL_GC_PUSH5(&phase_dims, &mag_dims, &jl_phase, &jl_TEs, &jl_mag);
        phase_dims->a = phase.shape(0);
        phase_dims->b = phase.shape(1);
        phase_dims->c = phase.shape(2);
        phase_dims->d = phase.shape(3);
        mag_dims->a = mag.shape(0);
        mag_dims->b = mag.shape(1);
        mag_dims->c = mag.shape(2);
        mag_dims->d = mag.shape(3);

        // Get data as julia arrays
        jl_phase =
            jl_ptr_to_array(jl_array4d, const_cast<T*>(phase.data()), reinterpret_cast<jl_value_t*>(phase_dims), 0);
        jl_TEs = jl_ptr_to_array_1d(jl_vector, const_cast<T*>(TEs.data()), TEs.size(), 0);
        jl_mag = jl_ptr_to_array(jl_array4d, const_cast<T*>(mag.data()), reinterpret_cast<jl_value_t*>(mag_dims), 0);

        // run voxel quality function
        jl_value_t* args[3] = {reinterpret_cast<jl_value_t*>(jl_phase), reinterpret_cast<jl_value_t*>(jl_TEs),
                               reinterpret_cast<jl_value_t*>(jl_mag)};
        jl_value_t* jl_vq_map = jl_call(jl_voxelquality, args, 3);
        // Capture any Julia exceptions and throw runtime error
        // TODO: see https://discourse.julialang.org/t/julia-exceptions-in-c/18387/2
        if (jl_exception_occurred()) {
            jl_value_t* exception = jl_exception_occurred();
            jl_value_t* sprint_fun = jl_get_function(jl_main_module, "sprint");
            jl_value_t* showerror_fun = jl_get_function(jl_main_module, "showerror");
            const char* returned_exception = jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exception));
            printf("ERROR: %s\n", returned_exception);
            throw std::runtime_error(jl_typeof_str(exception));
        }
        auto vq_map_ptr = static_cast<T*>(jl_array_data(jl_vq_map));

        // copy julia array to c++ vector
        std::vector<T> vq_map_vec(vq_map_ptr, vq_map_ptr + (phase.shape(0) * phase.shape(1) * phase.shape(2)));

        // pop arrays from root set
        JL_GC_POP();

        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        // return unwrapped phase
        return as_pyarray(std::move(vq_map_vec), {phase.shape(0), phase.shape(1), phase.shape(2)});
    }

    /**
     * @brief Destroy the Julia Context object
     *
     */
    ~JuliaContext() {
        // close Julia context
        jl_atexit_hook(0);
    }

    /**
     * @brief Wrapper for ROMEO unwrap function (3D)
     *
     * @param phase
     * @param weights
     * @param mag
     * @param mask
     * @param correctglobal
     * @param maxseeds
     * @param merge_regions
     * @param correct_regions
     * @return py::array_t<T, py::array::f_style>
     */
    py::array_t<T, py::array::f_style> romeo_unwrap3D(py::array_t<T, py::array::f_style> phase, std::string weights,
                                                      py::array_t<T, py::array::f_style> mag,
                                                      py::array_t<bool, py::array::f_style> mask,
                                                      bool correctglobal = false, int maxseeds = 1,
                                                      bool merge_regions = false, bool correct_regions = false) {
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();

        // Get dimensions as julia tuples
        jl_ntuple3_t* phase_dims = reinterpret_cast<jl_ntuple3_t*>(jl_new_struct_uninit(jl_ntuple3));
        jl_ntuple3_t* mag_dims = reinterpret_cast<jl_ntuple3_t*>(jl_new_struct_uninit(jl_ntuple3));
        jl_ntuple3_t* mask_dims = reinterpret_cast<jl_ntuple3_t*>(jl_new_struct_uninit(jl_ntuple3));
        jl_array_t* jl_phase;
        jl_array_t* jl_mag;
        jl_array_t* jl_mask;
        JL_GC_PUSH6(&phase_dims, &mag_dims, &mask_dims, &jl_phase, &jl_mag, &jl_mask);
        phase_dims->a = phase.shape(0);
        phase_dims->b = phase.shape(1);
        phase_dims->c = phase.shape(2);
        mag_dims->a = mag.shape(0);
        mag_dims->b = mag.shape(1);
        mag_dims->c = mag.shape(2);
        mask_dims->a = mask.shape(0);
        mask_dims->b = mask.shape(1);
        mask_dims->c = mask.shape(2);

        // Get data as julia arrays
        jl_phase =
            jl_ptr_to_array(jl_array3d, const_cast<T*>(phase.data()), reinterpret_cast<jl_value_t*>(phase_dims), 0);
        jl_mag = jl_ptr_to_array(jl_array3d, const_cast<T*>(mag.data()), reinterpret_cast<jl_value_t*>(mag_dims), 0);
        jl_mask = jl_ptr_to_array(jl_array3d_bool, const_cast<bool*>(mask.data()),
                                  reinterpret_cast<jl_value_t*>(mask_dims), 0);

        // get weights symbol
        auto jl_weights = string_to_symbol(weights);

        // get maxseeds
        auto jl_maxseeds = jl_box_int64(maxseeds);

        // get boolean values
        auto jl_correctglobal = correctglobal ? jl_true : jl_false;
        auto jl_merge_regions = merge_regions ? jl_true : jl_false;
        auto jl_correct_regions = correct_regions ? jl_true : jl_false;

        // run unwrap_individual
        jl_value_t* args[8] = {
            reinterpret_cast<jl_value_t*>(jl_phase),         reinterpret_cast<jl_value_t*>(jl_weights),
            reinterpret_cast<jl_value_t*>(jl_mag),           reinterpret_cast<jl_value_t*>(jl_mask),
            reinterpret_cast<jl_value_t*>(jl_correctglobal), reinterpret_cast<jl_value_t*>(jl_maxseeds),
            reinterpret_cast<jl_value_t*>(jl_merge_regions), reinterpret_cast<jl_value_t*>(jl_correct_regions)};
        jl_value_t* jl_unwrapped = jl_call(jl_unwrap3D, args, 8);
        // Capture any Julia exceptions and throw runtime error
        // TODO: see https://discourse.julialang.org/t/julia-exceptions-in-c/18387/2
        if (jl_exception_occurred()) {
            jl_value_t* exception = jl_exception_occurred();
            jl_value_t* sprint_fun = jl_get_function(jl_main_module, "sprint");
            jl_value_t* showerror_fun = jl_get_function(jl_main_module, "showerror");
            const char* returned_exception = jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exception));
            printf("ERROR: %s\n", returned_exception);
            throw std::runtime_error(jl_typeof_str(exception));
        }
        auto unwrapped_ptr = static_cast<T*>(jl_array_data(jl_unwrapped));

        // copy julia array to c++ vector
        std::vector<T> unwrapped_vec(unwrapped_ptr, unwrapped_ptr + phase.size());

        // pop arrays from root set
        JL_GC_POP();

        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        // return unwrapped phase
        return as_pyarray(std::move(unwrapped_vec), {phase.shape(0), phase.shape(1), phase.shape(2)});
    }

    /**
     * @brief Wrapper for ROMEO unwrap function (4D)
     *
     * @param phase
     * @param TEs
     * @param weights
     * @param mag
     * @param mask
     * @param correctglobal
     * @param maxseeds
     * @param merge_regions
     * @param correct_regions
     * @return py::array_t<T, py::array::f_style>
     */
    py::array_t<T, py::array::f_style> romeo_unwrap4D(py::array_t<T, py::array::f_style> phase,
                                                      py::array_t<T, py::array::f_style> TEs, std::string weights,
                                                      py::array_t<T, py::array::f_style> mag,
                                                      py::array_t<bool, py::array::f_style> mask,
                                                      bool correctglobal = false, int maxseeds = 1,
                                                      bool merge_regions = false, bool correct_regions = false) {
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();

        // Get dimensions as julia tuples
        jl_ntuple4_t* phase_dims = reinterpret_cast<jl_ntuple4_t*>(jl_new_struct_uninit(jl_ntuple4));
        jl_ntuple4_t* mag_dims = reinterpret_cast<jl_ntuple4_t*>(jl_new_struct_uninit(jl_ntuple4));
        jl_ntuple3_t* mask_dims = reinterpret_cast<jl_ntuple3_t*>(jl_new_struct_uninit(jl_ntuple3));
        jl_array_t* jl_phase;
        jl_array_t* jl_TEs;
        jl_array_t* jl_mag;
        jl_array_t* jl_mask;
        JL_GC_PUSH7(&phase_dims, &mag_dims, &mask_dims, &jl_phase, &jl_TEs, &jl_mag, &jl_mask);
        phase_dims->a = phase.shape(0);
        phase_dims->b = phase.shape(1);
        phase_dims->c = phase.shape(2);
        phase_dims->d = phase.shape(3);
        mag_dims->a = mag.shape(0);
        mag_dims->b = mag.shape(1);
        mag_dims->c = mag.shape(2);
        mag_dims->d = mag.shape(3);
        mask_dims->a = mask.shape(0);
        mask_dims->b = mask.shape(1);
        mask_dims->c = mask.shape(2);

        // Get data as julia arrays
        jl_phase =
            jl_ptr_to_array(jl_array4d, const_cast<T*>(phase.data()), reinterpret_cast<jl_value_t*>(phase_dims), 0);
        jl_TEs = jl_ptr_to_array_1d(jl_vector, const_cast<T*>(TEs.data()), TEs.size(), 0);
        jl_mag = jl_ptr_to_array(jl_array4d, const_cast<T*>(mag.data()), reinterpret_cast<jl_value_t*>(mag_dims), 0);
        jl_mask = jl_ptr_to_array(jl_array3d_bool, const_cast<bool*>(mask.data()),
                                  reinterpret_cast<jl_value_t*>(mask_dims), 0);

        // get weights symbol
        auto jl_weights = string_to_symbol(weights);

        // get maxseeds
        auto jl_maxseeds = jl_box_int64(maxseeds);

        // get boolean values
        auto jl_correctglobal = correctglobal ? jl_true : jl_false;
        auto jl_merge_regions = merge_regions ? jl_true : jl_false;
        auto jl_correct_regions = correct_regions ? jl_true : jl_false;

        // run unwrap_individual
        jl_value_t* args[9] = {
            reinterpret_cast<jl_value_t*>(jl_phase),          reinterpret_cast<jl_value_t*>(jl_TEs),
            reinterpret_cast<jl_value_t*>(jl_weights),        reinterpret_cast<jl_value_t*>(jl_mag),
            reinterpret_cast<jl_value_t*>(jl_mask),           reinterpret_cast<jl_value_t*>(jl_correctglobal),
            reinterpret_cast<jl_value_t*>(jl_maxseeds),       reinterpret_cast<jl_value_t*>(jl_merge_regions),
            reinterpret_cast<jl_value_t*>(jl_correct_regions)};
        jl_value_t* jl_unwrapped = jl_call(jl_unwrap4D, args, 9);
        // Capture any Julia exceptions and throw runtime error
        // TODO: see https://discourse.julialang.org/t/julia-exceptions-in-c/18387/2
        if (jl_exception_occurred()) {
            jl_value_t* exception = jl_exception_occurred();
            jl_value_t* sprint_fun = jl_get_function(jl_main_module, "sprint");
            jl_value_t* showerror_fun = jl_get_function(jl_main_module, "showerror");
            const char* returned_exception = jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exception));
            printf("ERROR: %s\n", returned_exception);
            throw std::runtime_error(jl_typeof_str(exception));
        }
        auto unwrapped_ptr = static_cast<T*>(jl_array_data(jl_unwrapped));

        // copy julia array to c++ vector
        std::vector<T> unwrapped_vec(unwrapped_ptr, unwrapped_ptr + phase.size());

        // pop arrays from root set
        JL_GC_POP();

        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        // return unwrapped phase
        return as_pyarray(std::move(unwrapped_vec), {phase.shape(0), phase.shape(1), phase.shape(2), phase.shape(3)});
    }

   private:
    jl_function_t* jl_voxelquality;
    jl_function_t* jl_unwrap3D;
    jl_function_t* jl_unwrap4D;
    jl_tupletype_t* jl_ntuple3;
    jl_tupletype_t* jl_ntuple4;
    jl_value_t* jl_vector;
    jl_value_t* jl_array3d_bool;
    jl_value_t* jl_array3d;
    jl_value_t* jl_array4d;

    typedef struct {
        py::ssize_t a;
        py::ssize_t b;
        py::ssize_t c;
    } jl_ntuple3_t;

    typedef struct {
        py::ssize_t a;
        py::ssize_t b;
        py::ssize_t c;
        py::ssize_t d;
    } jl_ntuple4_t;

    /**
     * @brief Convert a string to julia symbol
     *
     * @param str
     * @return jl_value_t*
     */
    jl_value_t* string_to_symbol(std::string str) {
        std::ostringstream stringStream;
        stringStream << ":" << str;
        return jl_eval_string(stringStream.str().c_str());
    }
};

#endif
