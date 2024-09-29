#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_global_atomic(__global const unsigned* arr, unsigned size, __global unsigned* result) {
    unsigned id = get_global_id(0);

    if (id < size) {
        atomic_add(result, arr[id]);
    }
}
__kernel void sum_loop(__global const unsigned* arr, unsigned size, __global unsigned* result) {
#define VALS_PER_WORKITEM 64
    unsigned id = get_global_id(0);

    unsigned res = 0;
    for (int i = 0; i < VALS_PER_WORKITEM; ++i) {
        int idx = id * VALS_PER_WORKITEM + i;
        if (idx < size) {
            res += arr[idx];
        }
    }

    atomic_add(result, res);
#undef VALS_PER_WORKITEM
}
__kernel void sum_loop_coalesced(__global const unsigned* arr, unsigned size, __global unsigned* result) {
#define VALS_PER_WORKITEM 64
    const unsigned localId = get_local_id(0);
    const unsigned groupId = get_group_id(0);
    const unsigned localSize = get_local_size(0);

    unsigned res = 0;
    for (int i = 0; i < VALS_PER_WORKITEM; ++i) {
        // possible overflow
        unsigned idx = groupId * localSize * VALS_PER_WORKITEM + i * localSize + localId;
        if (idx < size) {
            res += arr[idx];
        }
    }
    atomic_add(result, res);
#undef VALS_PER_WORKITEM
}
__kernel void sum_local_mem(__global const unsigned* arr, unsigned size, __global unsigned* result) {
#define WORKGROUP_SIZE 128
    unsigned globalId = get_global_id(0);
    unsigned localId = get_local_id(0);

    __local unsigned localArr[WORKGROUP_SIZE];
    localArr[localId] = globalId < size ? arr[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned partialSum = 0;
        for (unsigned i = 0; i < WORKGROUP_SIZE; ++i) {
            partialSum += localArr[i];
        }
        atomic_add(result, partialSum);
    }
#undef WORKGROUP_SIZE
}
__kernel void sum_tree(__global const unsigned* arr, unsigned size, __global unsigned* result) {
#define WORKGROUP_SIZE 128
    unsigned globalId = get_global_id(0);
    unsigned localId = get_local_id(0);

    __local unsigned localArr[WORKGROUP_SIZE];
    localArr[localId] = globalId < size ? arr[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned k = WORKGROUP_SIZE; k > 1; k /= 2) {
        if (localId * 2 < k) {
            int a = localArr[localId];
            int b = localArr[localId + k / 2];
            localArr[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(result, localArr[0]);
    }

#undef WORKGROUP_SIZE
}