#define WORK_GROUP_SIZE 8
#define BITS_FOR_SORT 4

__kernel void clear(
        __global unsigned int *as,
        unsigned int n
) {
    const unsigned int global_id = get_global_id(0);

    if (global_id >= n)
        return;

    as[global_id] = 0;
}

__kernel void count(
        __global const unsigned int *as,
        __global unsigned int *counters,
        const unsigned int n,
        const unsigned int shift
) {
    const unsigned int global_id = get_global_id(0);

    if (global_id >= n)
        return;

    const unsigned int group_id = get_group_id(0);
    const unsigned int local_size = get_local_size(0);

    atomic_inc(&counters[group_id * (1 << BITS_FOR_SORT) + ((as[global_id] >> shift) & ((1 << BITS_FOR_SORT) - 1))]);
}

__kernel void radix_sort(
        __global const unsigned int *as,
        __global unsigned int *bs,
        __global const unsigned int *counters_t,
        unsigned int n,
        unsigned int shift
) {
    const unsigned int global_id = get_global_id(0);

    if (global_id >= n)
        return;

    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int groups_num = get_num_groups(0);

    __local unsigned int counter_is[WORK_GROUP_SIZE];
    counter_is[local_id] = (as[global_id] >> shift) & ((1 << BITS_FOR_SORT) - 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int counters_idx = counter_is[local_id] * groups_num + group_id;
    unsigned int smaller_outside_wg = 0;
    if (counters_idx > 0)
        smaller_outside_wg = counters_t[counters_idx - 1];

    unsigned int same_inside_wg = 0;
    for (int i = 0; i < local_id; ++i)
        if (counter_is[i] == counter_is[local_id])
            same_inside_wg++;

    bs[smaller_outside_wg + same_inside_wg] = as[global_id];
}

#define TILE_SIZE 16

__kernel void matrix_transpose(
        __global float *as,
        __global float *as_t,
        const unsigned int M,
        const unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    unsigned int group_i = get_group_id(0);
    unsigned int group_j = get_group_id(1);

    unsigned int i_new = group_i * TILE_SIZE + local_j;
    unsigned int j_new = group_j * TILE_SIZE + local_i;

    tile[local_j][local_i] = as[j * M + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[i_new * K + j_new] = tile[local_i][local_j];
}

// prefix_sum
__kernel void up_sweep(
        __global unsigned int *as,
        unsigned int sum_len,
        unsigned int n
) {
    const unsigned int i = get_global_id(0);

    if (i * sum_len + sum_len - 1 >= n)
        return;

    as[i * sum_len + sum_len - 1] += as[i * sum_len + sum_len / 2 - 1];
}

__kernel void down_sweep(
        __global unsigned int *as,
        unsigned int sum_len,
        unsigned int n
) {
    // we don't need the first element there
    const unsigned int i = get_global_id(0) + 1;

    if (i * sum_len + sum_len / 2 - 1 >= n)
        return;

    as[i * sum_len + sum_len / 2 - 1] += as[i * sum_len - 1];
}