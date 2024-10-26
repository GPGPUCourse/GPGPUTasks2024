#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6
__kernel void prefix_stage1(__global unsigned int *as, unsigned int step, unsigned int n) {
    int global_id = get_global_id(0);

    int arr_id = (global_id + 1) * (1 << step) - 1;

    if (arr_id >= n)
        return;

    as[arr_id] += as[arr_id - (1 << (step - 1))];
}

#line 6
__kernel void prefix_stage2(__global unsigned int *as, unsigned int step, unsigned int n) {
    int global_id = get_global_id(0);

    int cur_block = (1 << step);

    int arr_id = (global_id + 1) * (1 << step) - 1;

    if (arr_id + cur_block / 2 >= n)
        return;

    as[arr_id + cur_block / 2] += as[arr_id];
}