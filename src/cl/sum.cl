#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum01_global_atomic(
    __global unsigned int *a,
    unsigned int n,
    __global unsigned int *result
) {
    int i = get_global_id(0);
    if (i < n) {
        atomic_add(result, a[i]);
    }
}

#define VALUES_PER_WORKITEM 64

__kernel void sum02_multiple_per_warp_non_coalesced(
        __global unsigned int *a,
        unsigned int n,
        __global unsigned int *result
) {
    int i = get_global_id(0);

    unsigned int sum = 0;
    for (int j = 0; j < VALUES_PER_WORKITEM; ++j) {
        int idx = i * VALUES_PER_WORKITEM + j;
        if (idx < n) {
            sum += a[idx];
        }
    }

    atomic_add(result, sum);
}

__kernel void sum03_multiple_per_warp_coalesced(
        __global unsigned int *a,
        unsigned int n,
        __global unsigned int *result
) {
    int i = get_global_id(0);

    unsigned int sum = 0;
    for (int j = 0; j < n; j += get_global_size(0)) {
        int idx = i + j;
        if (idx < n) {
            sum += a[idx];
        }
    }

    atomic_add(result, sum);
}

#define WORKGROUP_SIZE 128

__kernel void sum04_local_memory_tree(
        __global unsigned int *a,
        unsigned int n,
        __global unsigned int *result
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int two_lid = lid << 1;
    int two_gid = gid << 1;

    __local unsigned int buf[WORKGROUP_SIZE * 2];

    int idx = get_group_id(0) * (WORKGROUP_SIZE * 2) + lid;
    if (idx < n) {
        buf[lid] = a[idx];
    } else {
        buf[lid] = 0;
    }

    idx += WORKGROUP_SIZE;
    if (idx < n) {
        buf[lid + WORKGROUP_SIZE] = a[idx];
    } else {
        buf[lid + WORKGROUP_SIZE] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WORKGROUP_SIZE; i > 0; i >>= 1) {
        if (lid < i) {
            buf[lid] += buf[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(result, buf[0]);
    }
}

__kernel void sum05_global_tree(
        __global unsigned int *a,
        __global unsigned int *b,
        unsigned int n,
        unsigned int half_n
) {
    int i = get_global_id(0);
    unsigned int sum = a[i];
    int j = i + half_n;
    if (j < n) {
        sum += a[j];
    }
    b[i] = sum;
}

__kernel void sum06_wider_global_tree(
        __global unsigned int *a,
        __global unsigned int *b,
        unsigned int n,
        unsigned int k
) {
    int idx = get_global_id(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        if (idx < n) {
            sum += a[idx];
        }
        idx += k;
    }
    b[get_global_id(0)] = sum;
}
