#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
// TODO
__kernel void sum_1(__global const unsigned int* array, __global unsigned int* sum, unsigned int n) {
    const unsigned global_id = get_global_id(0);
    if (global_id >= n)
        return;
    atomic_add(sum, array[global_id]);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_2(__global const unsigned int* array, __global unsigned int* sum, unsigned int n) {
    unsigned int result = 0;
    const unsigned global_id = get_global_id(0);
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = global_id * VALUES_PER_WORKITEM + i;
        if (index < n) {
            result += array[index];
        }
    }
    atomic_add(sum, result);
}

__kernel void sum_3(__global const unsigned int* array, __global unsigned int* sum, unsigned int n) {
    unsigned int result = 0;
    const unsigned local_id = get_local_id(0);
    const unsigned work_group_id = get_group_id(0);
    const unsigned local_size = get_local_size(0);
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = work_group_id * local_size * VALUES_PER_WORKITEM + i * local_size + local_id;
        if (index < n) {
            result += array[index];
        }
    }
    atomic_add(sum, result);
}

#define WORKGROUP_SIZE 128
__kernel void sum_4(__global const unsigned int* array, __global unsigned int* sum, unsigned int n) {
    const unsigned global_id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    __local unsigned int buffer[WORKGROUP_SIZE];
    if (global_id < n)
        buffer[local_id] = array[global_id];
    else
        buffer[local_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unsigned int result = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            result += buffer[i];
        }
        atomic_add(sum, result);
    }
}

__kernel void sum_5(__global const unsigned int* array, __global unsigned int* sum, unsigned int n) {
    const unsigned global_id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);
    __local unsigned int buffer[WORKGROUP_SIZE];
    if (global_id < n)
        buffer[local_id] = array[global_id];
    else
        buffer[local_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * local_id < nValues) {
            unsigned int a = buffer[local_id];
            unsigned int b = buffer[local_id + nValues / 2];
            buffer[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        sum[group_id] = buffer[0];
    }
}
