#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 128

__kernel void atomic_sum(
    __global const unsigned int *as,
    __global unsigned int *sum,
    unsigned int n
) {
    const unsigned int index = get_global_id(0);

    if (index >= n) {
        return;
    };

    atomic_add(sum, as[index]);
}

__kernel void cycle_sum(
    __global const unsigned int *as,
    __global unsigned int *sum,
    unsigned int n
) {
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        const unsigned int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += as[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void cycle_coalesced_sum(
    __global const unsigned int *as,
    __global unsigned int *sum,
    unsigned int n
) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        const unsigned int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            res += as[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void local_mem_sum(
    __global const unsigned int *as,
    __global unsigned int *sum,
    unsigned int n
) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = as[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void tree_sum(
    __global const unsigned int *as,
    __global unsigned int *sum,
    unsigned int n
) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}