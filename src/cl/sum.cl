#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_global_add(__global const unsigned int* arr,
                             __global unsigned int* sum,
                             unsigned int n) {
    const unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    atomic_add(sum, arr[gid]);
}

#define VALUES_PER_WORK_ITEM 32
__kernel void sum_loop(__global const unsigned int* arr,
                       __global unsigned int* sum,
                       unsigned int n) {
    const unsigned int gid = get_global_id(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        int idx = gid * VALUES_PER_WORK_ITEM + i;
        if (idx < n) {
            res += arr[idx];;
        }
    }

    atomic_add(sum, res);
}

__kernel void coalesced(__global const unsigned int* arr,
                        __global unsigned int* sum,
                        unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORK_ITEM + i * grs + lid;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

#define WORKGROUP_SIZE 128
__kernel void local_memory(__global const unsigned int* arr,
                           __global unsigned int* sum,
                           unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = arr[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void tree_sum(__global const unsigned int* arr,
                       __global unsigned int* sum,
                       unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n_vals = WORKGROUP_SIZE; n_vals > 1; n_vals /= 2) {
        if (2 * lid < n_vals) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + n_vals / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}





