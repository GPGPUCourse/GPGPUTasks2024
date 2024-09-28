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
        unsigned long long idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_cycle_coalesced(
        __global const unsigned int *arr,
        __global unsigned int *sum,
        unsigned int n) {
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);
    int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = work_group_id * group_size * VALUES_PER_WORKITEM + i * group_size + local_id;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

#define WORKGROUP_SIZE 128
__kernel void sum_one_main_thread(
        __global const unsigned int *arr,
        __global unsigned int *sum,
        unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[local_id] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

#define WORKGROUP_SIZE 128
__kernel void sum_tree(
        __global const unsigned int *arr,
        __global unsigned int *sum,
        unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[local_id] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int values_count = WORKGROUP_SIZE; values_count > 1; values_count /= 2) {
        if (local_id * 2 < values_count) {
            int a = buf[local_id];
            int b = buf[local_id + values_count / 2];
            buf[local_id] = a + b;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(sum, buf[0]);
    }

}

