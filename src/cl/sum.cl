__kernel void sum_1(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    int idx = get_global_id(0);

    if (idx >= n) {
        return;
    }

    atomic_add(sum, arr[idx]);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_2(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void sum_3(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);
    const unsigned int lid = get_local_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int work_per_group = VALUES_PER_WORKITEM * grs;
        int idx = wid * work_per_group + lid + i * grs;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

#define WORKGROUP_SIZE 32
__kernel void sum_4(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = arr[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int res = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++) {
            res += buf[i];
        }
        atomic_add(sum, res);
    }
}

__kernel void sum_5(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = arr[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i = lid;
    int j = lid + 1;
    while (j < WORKGROUP_SIZE) {
        buf[i] += buf[j];
        i *= 2;
        j *= 2;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}
