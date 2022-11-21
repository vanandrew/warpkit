#include <romeo.h>
#include <julia.h>
#include <stdio.h>

int jl_unwrap_individual() {
    // initialize Julia context
    jl_init();

    // load MriResearchTools
    jl_eval_string("using MriResearchTools;");
    if (jl_exception_occurred())
        printf("0: %s \n", jl_typeof_str(jl_exception_occurred()));
    
    // Get functions
    jl_eval_string("unwrap_individual_pos(phase, TEs) = unwrap_individual(phase, TEs=TEs);");
    if (jl_exception_occurred())
        printf("1: %s \n", jl_typeof_str(jl_exception_occurred()));
    jl_function_t *jl_unwrap_individual = (jl_function_t*) jl_eval_string("unwrap_individual_pos;");
    if (jl_exception_occurred())
        printf("2: %s \n", jl_typeof_str(jl_exception_occurred()));
    jl_eval_string("savenii_pos(data, filename, header_data) = savenii(data, filename, header=header(header_data));");
    if (jl_exception_occurred())
        printf("3: %s \n", jl_typeof_str(jl_exception_occurred()));
    jl_function_t *jl_savenii = (jl_function_t*) jl_eval_string("savenii_pos;");
    if (jl_exception_occurred())
        printf("4: %s \n", jl_typeof_str(jl_exception_occurred()));

    jl_value_t **root;
    JL_GC_PUSHARGS(root, 4);

    // Get phase data and TEs
    root[0] = jl_eval_string("readphase(\"/home/vanandrew/Data/combined.nii\");");
    if (jl_exception_occurred())
        printf("5: %s \n", jl_typeof_str(jl_exception_occurred()));
    root[1] = jl_eval_string("[14.2, 38.93, 63.66, 88.39, 113.12]");
    if (jl_exception_occurred())
        printf("6: %s \n", jl_typeof_str(jl_exception_occurred()));
    
    // Get unwrapped phase
    root[2] = jl_call2(jl_unwrap_individual, root[0], root[1]);
    if (jl_exception_occurred())
        printf("7: %s \n", jl_typeof_str(jl_exception_occurred()));
    
    // Save the data
    root[3] = jl_eval_string("\"/home/vanandrew/Data/unwrapped.nii\"");
    if (jl_exception_occurred())
        printf("8: %s \n", jl_typeof_str(jl_exception_occurred()));
    jl_call3(jl_savenii, root[2], root[3], root[0]);
    if (jl_exception_occurred())
        printf("9: %s \n", jl_typeof_str(jl_exception_occurred()));

    JL_GC_POP();
    return 0;
}
