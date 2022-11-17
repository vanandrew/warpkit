#include <julia.h>
#include <stdio.h>

int main() {
    // initialize Julia context
    jl_init();
    // // Create array
    // jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    // jl_array_t* x = jl_alloc_array_1d(array_type, 3);
    // JL_GC_PUSH1(&x);
    // for (int i = 0; i < 3; i++) jl_arrayset(x, jl_box_float64(i), i);
    // // get array data
    // double* x_data = (double*)jl_array_data(x);
    // // print array data
    // for (int i = 0; i < 3; i++) printf("%f\n", x_data[i]);
    // JL_GC_POP();
    jl_eval_string("using MriResearchTools");
    jl_eval_string("phase = readphase(\"/home/vanandrew/combined.nii\");");
    jl_eval_string("unwrapped = unwrap_individual(phase, TEs=[14.2, 38.93, 63.66, 88.39, 113.12]);");
    jl_eval_string("savenii(unwrapped, \"/home/vanandrew/unwrapped.nii\", header=header(phase));");
    return 0;
}
