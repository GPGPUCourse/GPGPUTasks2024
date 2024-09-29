// 3.2.1 Суммирование с глобальным атомарным добавлением (просто как бейзлайн)
__kernel void sum_baseline(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    const unsigned int N
)
{
    const unsigned int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    atomic_add(sum, arr[gid]);
}



// 3.2.2 Суммирование с циклом
#define VALUES_PER_WORKITEM (64)
__kernel void sum_cycle(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    const unsigned int N
)
{
    const unsigned int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    unsigned int res = 0;

    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; i++) {
        const unsigned int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < N) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

// 3.2.3 Суммирование с циклом и coalesced доступом (интересно сравнение по скорости с не-coalesced версией)
__kernel void sum_cycle_coalesced(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    const unsigned int N
)
{
    const unsigned int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;

    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; i++) {
        const unsigned int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < N) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

// 3.2.4 Суммирование с локальной памятью и главным потоком
#define WORKGROUP_SIZE (128)
__kernel void sum_local_memory(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    const unsigned int N
)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < N ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; i++) {
            group_res += buf[i];
        }
    
        atomic_add(sum, group_res);
    }
}

// 3.2.5 Суммирование с деревом
__kernel void sum_tree(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    const unsigned int N
)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < N ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
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