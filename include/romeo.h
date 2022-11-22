#ifndef ROMEO_H
#define ROMEO_H

#include <julia.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <utilities.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

/**
 * @brief Object to manage Julia Context
 *
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
        jl_vector = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 1);
        jl_array3d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 3);
        jl_array4d = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_float64_type), 4);

        // initialize module
        jl_eval_string("using ROMEO;");

        // get functions from modules
        jl_unwrap_individual = static_cast<jl_function_t*>(
            jl_eval_string("unwrap_individual_positional_wrapper(phase, TEs, weights, mag, mask, correctglobal) = "
                           "unwrap_individual(phase, TEs=TEs, mag=mag, mask=mask, correctglobal=correctglobal);"));
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
    py::array_t<T, py::array::f_style> romeo_unwrap_individual(
        py::array_t<T, py::array::f_style> phase, py::array_t<T, py::array::f_style> TEs, std::string weights,
        py::array_t<T, py::array::f_style> mag, py::array_t<T, py::array::f_style> mask, bool correctglobal = false) {
        jl_function_t* println = jl_get_function(jl_base_module, "println");
        jl_function_t* display = jl_get_function(jl_base_module, "display");
        jl_function_t* type_of = static_cast<jl_function_t*>(jl_eval_string("x -> println(typeof(x))"));
        std::cout << "\n";

        // Get dimensions
        // jl_array_t* phase_dims =
        //     jl_ptr_to_array_1d(jl_size_vector, const_cast<ssize_t*>(phase.shape()), phase.ndim(), 0);
        // jl_array_t* mag_dims = jl_ptr_to_array_1d(jl_size_vector, const_cast<ssize_t*>(mag.shape()), mag.ndim(), 0);
        // jl_array_t* mask_dims = jl_ptr_to_array_1d(jl_size_vector, const_cast<ssize_t*>(mask.shape()), mask.ndim(), 0);


        // Get data as julia arrays
        jl_value_t* dim = jl_eval_string("(110, 110, 72, 5)");
        jl_array_t* jl_phase =
            jl_ptr_to_array(jl_array4d, const_cast<T*>(phase.data()), dim, 0);
        jl_array_t* jl_TEs = jl_ptr_to_array_1d(jl_vector, const_cast<T*>(TEs.data()), TEs.size(), 0);
        // jl_array_t* jl_mag =
        //     jl_ptr_to_array(jl_array4d, const_cast<T*>(mag.data()), reinterpret_cast<jl_value_t*>(mag_dims), 0);
        // jl_array_t* jl_mask =
        //     jl_ptr_to_array(jl_array3d, const_cast<T*>(mask.data()), reinterpret_cast<jl_value_t*>(mask_dims), 0);

        // push root set for arrays
        JL_GC_PUSHARGS(root_set, 8);
        // root_set[0] = reinterpret_cast<jl_value_t*>(phase_dims);
        // root_set[1] = reinterpret_cast<jl_value_t*>(mag_dims);
        // root_set[2] = reinterpret_cast<jl_value_t*>(mask_dims);
        root_set[3] = reinterpret_cast<jl_value_t*>(jl_phase);
        root_set[4] = reinterpret_cast<jl_value_t*>(jl_TEs);
        // root_set[5] = reinterpret_cast<jl_value_t*>(jl_mag);
        // root_set[6] = reinterpret_cast<jl_value_t*>(jl_mask);

        // jl_call1(println, (jl_value_t*)phase_dims);
        // jl_call1(println, (jl_value_t*)mag_dims);
        // jl_call1(println, (jl_value_t*)mask_dims);
        // jl_call1(display, (jl_value_t*)jl_phase);
        jl_call1(type_of, (jl_value_t*)jl_phase);
        jl_call1(println, (jl_value_t*)jl_TEs);
        // jl_call1(display, (jl_value_t*)jl_mag);
        // jl_call1(display, (jl_value_t*)jl_mask);

        // get weights symbol
        auto weights_sym = string_to_symbol(weights);

        // get boolean value
        auto correctglobal_val = correctglobal ? jl_true : jl_false;

        // test
        jl_call1(println, jl_unwrap_individual);
        jl_call1(type_of, jl_unwrap_individual);
        jl_call1(println, weights_sym);
        jl_call1(type_of, weights_sym);
        jl_call1(println, correctglobal_val);
        jl_call1(type_of, correctglobal_val);

        // pop the root set
        JL_GC_POP();
        return std::move(phase);
    }

   private:
    jl_function_t* jl_unwrap_individual;
    jl_value_t* jl_vector;
    jl_value_t* jl_array3d;
    jl_value_t* jl_array4d;
    jl_value_t** root_set;

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
    };
};

#endif