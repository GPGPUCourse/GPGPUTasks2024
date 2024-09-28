#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void sum_atomic(
        __global const unsigned int *arr,
        __global unsigned int *sum,
        unsigned int n) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n) {
        return;
    };

    atomic_add(sum, arr[gid]);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_cycle(
        __global const unsigned int *arr,
        __global unsigned int *sum,
        unsigned int n) {
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        unsigned int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}
