#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void global_atomic_add(__global const unsigned int* as,
                                    __global unsigned int* sum,
                                    unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    atomic_add(sum, as[gid]);
}

#define VALUES_PER_WORKITEM 64
__kernel void cycled(__global const unsigned int* as,
                     __global unsigned int* sum,
                     unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += as[idx];
        }
    }

    atomic_add(sum, res);
}
#define VALUES_PER_WORKITEM 64
__kernel void cycled_coalesced(__global const unsigned int* as,
                     __global unsigned int* sum,
                     unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            res += as[idx];
        }
    }

    atomic_add(sum, res);
}
#define WORKGROUP_SIZE 128
__kernel void local_mem_with_main_thread(__global const unsigned int* as,
                     __global unsigned int* sum,
                     unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = as[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}
#define WORKGROUP_SIZE 128
__kernel void tree(__global const unsigned int* as,
                     __global unsigned int* sum,
                     unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORKGROUP_SIZE; i > 1; i /=2) {
        if (lid * 2 < i) {
            buf[lid] += buf[lid + i / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}
