#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global const unsigned int *as, unsigned int n, __global unsigned int *s)
{
#ifdef VARIANT0
    const unsigned int index = get_global_id(0);

    if (index >= n) return;

    atomic_add(s, as[index]);
#endif

#ifdef VARIANT1
    const unsigned int index = get_global_id(0);

    unsigned int sum = 0;
    for (int idx = index * WORK_PER_ITEM; idx < (index + 1) * WORK_PER_ITEM; ++idx) {
        if (idx >= n) break;
        sum += as[idx];
    }
    atomic_add(s, sum);
#endif

#ifdef VARIANT2
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int sum = 0;
    for (int idx = gid * grs * WORK_PER_ITEM + lid; idx < (gid + 1) * grs * WORK_PER_ITEM; idx += grs) {
        if (idx >= n) break;
        sum += as[idx];
    }
    atomic_add(s, sum);
#endif

#ifdef VARIANT3
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid != 0) return;

    unsigned int sum = 0;
    for (int idx = 0; idx < WORKGROUP_SIZE; ++idx) {
        sum += buf[idx];
    }
    atomic_add(s, sum);
#endif
}

__kernel void tree_sum(__global const unsigned int *in, unsigned int n, __global unsigned int *out)
{
    const unsigned int gid = get_global_id(0);

    const unsigned int idxOut = gid;
    const unsigned int stride = (n + 1) / 2;

    if (gid * 2 >= n) return;
    if (gid * 2 + 1 >= n) {
        out[idxOut] = in[gid];
    } else {
        out[idxOut] = in[gid] + in[gid + stride];
    }
}

__kernel void tree_sum2(__global const unsigned int *in, unsigned int n, __global unsigned int *out)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_group_id(0);
    const unsigned int wid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE * 2];
    __local unsigned int *buf1 = buf, *buf2 = buf + WORKGROUP_SIZE;

    buf1[lid] = wid < n ? in[wid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            buf2[lid] = buf1[lid] + buf1[lid + stride];
        }
        __local unsigned int *tmp = buf1; buf1 = buf2; buf2 = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[gid] = buf1[0];
    }
}
