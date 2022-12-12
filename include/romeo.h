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
 * @tparam jl_t
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
        if (std::is_same<T, double>::value)
            jl_vector = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 1);
        else if (std::is_same<T, float>::value)
            jl_vector = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float32_type), 1);
        jl_array3d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_bool_type), 3);
        if (std::is_same<T, double>::value)
            jl_array4d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 4);
        else if (std::is_same<T, float>::value)
            jl_array4d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float32_type), 4);

        // initialize module
        jl_eval_string("using ROMEO;");

        // get functions from modules
        jl_unwrap_individual = static_cast<jl_function_t*>(jl_eval_string(
            "unwrap_individual_positional_wrapper(phase, TEs, weights, mag, mask, correctglobal) = "
            "unwrap_individual(phase, TEs=TEs, weights=:romeo3, mag=mag, mask=mask, correctglobal=correctglobal);"));
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
     * @brief Wrapper for ROMEO unwrap_individual function
     *
     * @param phase
     * @param TEs
     * @param weights
     * @param mag
     * @param mask
     * @param correctglobal
     * @return py::array_t<T, py::array::f_style>
     */
    py::array_t<T, py::array::f_style> romeo_unwrap_individual(py::array_t<T, py::array::f_style> phase,
                                                               py::array_t<T, py::array::f_style> TEs,
                                                               std::string weights,
                                                               py::array_t<T, py::array::f_style> mag,
                                                               py::array_t<bool, py::array::f_style> mask,
                                                               bool correctglobal = false) {
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
        jl_mask =
            jl_ptr_to_array(jl_array3d, const_cast<bool*>(mask.data()), reinterpret_cast<jl_value_t*>(mask_dims), 0);

        // get weights symbol
        auto jl_weights = string_to_symbol(weights);

        // get boolean value
        auto jl_correctglobal = correctglobal ? jl_true : jl_false;

        // run unwrap_individual
        jl_value_t* args[6] = {
            reinterpret_cast<jl_value_t*>(jl_phase),   reinterpret_cast<jl_value_t*>(jl_TEs),
            reinterpret_cast<jl_value_t*>(jl_weights), reinterpret_cast<jl_value_t*>(jl_mag),
            reinterpret_cast<jl_value_t*>(jl_mask),    reinterpret_cast<jl_value_t*>(jl_correctglobal)};
        // std::cout << "Unwrapping..." << std::endl;
        jl_value_t* jl_unwrapped = jl_call(jl_unwrap_individual, args, 6);
        // std::cout << "Unwrapping complete." << std::endl;
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
    jl_function_t* jl_unwrap_individual;
    jl_tupletype_t* jl_ntuple3;
    jl_tupletype_t* jl_ntuple4;
    jl_value_t* jl_vector;
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
