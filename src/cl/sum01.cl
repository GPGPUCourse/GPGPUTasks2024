#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void sum_kernel(__global const uint *data, __global uint *result, uint n) {
    uint global_id = get_global_id(0);
    if (global_id < n) {
        atomic_add(result, data[global_id]);
    }
}