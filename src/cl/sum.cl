#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 128
#define VALUES_PER_WORKITEM 64

__kernel void global_add_scan(__global const unsigned int *array,
                              __global unsigned int *sum,
                              unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    if (gid >= n) {
        return;
    }

    atomic_add(sum, array[gid]);
}

__kernel void simple_cycle_scan(__global unsigned int *array, 
                                __global unsigned int *sum, 
                                unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    unsigned int result = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * VALUES_PER_WORKITEM + i;

        if (idx < n) {
            result += array[idx];
        }
    }

    atomic_add(sum, result);
}

__kernel void coalesced_cycle_scan(__global unsigned int *array, 
                                   __global unsigned int *sum, 
                                   unsigned int n)
{
    const unsigned int group_size = get_local_size(0);
    const unsigned int gid = get_group_id(0);
    const unsigned int lid = get_local_id(0);

    int result = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * group_size * VALUES_PER_WORKITEM + i * group_size + lid;

        if (idx < n) {
            result += array[idx];
        }
    }

    atomic_add(sum, result);
}

__kernel void local_leader_synchronization_scan(__global const unsigned int *array,
                                                __global unsigned int *sum,
                                                unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    buffer[lid] = array[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0) {
        return;
    }

    unsigned int result = 0;
    for (int i = 0; i < WORKGROUP_SIZE; i++) {
        result += buffer[i];
    }

    atomic_add(sum, result);
}

__kernel void divide_and_conquer_scan(__global const unsigned int *array,
                                      __global unsigned int *sum,
                                      unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    if (gid < n) {
        buffer[lid] = array[gid];
    } else {
        buffer[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORKGROUP_SIZE; i > 1; i /= 2) {
        if (lid * 2 < i) {
            buffer[lid] += buffer[lid + i / 2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buffer[0]);
    }
}
