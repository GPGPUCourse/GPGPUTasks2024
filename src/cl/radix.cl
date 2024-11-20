unsigned int helper(unsigned int value, unsigned int bits_number, unsigned int bits_shift) {
    return ((1 << bits_number) - 1) & (value >> bits_shift);
}

__kernel void clear(__global unsigned int *as)
{
    int gid = get_global_id(0);
    as[gid] = 0;
}

__kernel void radix_numbers(__global const unsigned int *as, 
                            __global unsigned int *counters, 
                            unsigned int n, 
                            unsigned int shift)
{
    int gid = get_global_id(0);
    int group_id = get_group_id(0);

    int k = helper(as[gid], shift, n);
    atomic_inc(&counters[group_id * (1 << shift) + k]);
}

__kernel void prefix_sum_down(__global unsigned int* as,
                            const unsigned int step, 
                            const unsigned int n) {
    const uint gid = get_global_id(0);
    int index = 2 * step * (gid + 1) - 1;
    if (index < n) {
        as[index] += as[index - step];
    }
}

__kernel void prefix_sum_up(__global unsigned int* as,
                              const unsigned int step, 
                              const unsigned int n) {
    const uint gid = get_global_id(0);
    int index =  step * (2 * gid + 3) - 1;
    if (index < n) {
        as[index] += as[index - step];
    }
}

__kernel void matrix_transpose_local_good_banks(__global unsigned int* matrix, 
                                                __global unsigned int* result, 
                                                const unsigned int width, 
                                                const unsigned int height)
{
    const unsigned int gidi = get_global_id(0);
    const unsigned int gidj = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int buffer_size = 16;

    __local unsigned int buffer[buffer_size][buffer_size + 1];

    if (gidj < height && gidi < width) {
        buffer[local_j][local_i] = matrix[gidj * width + gidi];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int x = gidj - local_j + local_i;
    const unsigned int y = gidi + local_j - local_i;

    if (y < width && x < height)
        result[y * height + x] = buffer[local_i][local_j];
}

__kernel void radix_sort(__global const unsigned int *as, 
                         __global unsigned int *bs,
                         __global const unsigned int *counters, 
                         const unsigned int pos, 
                         const unsigned int nbits)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int buffer[128];
    buffer[local_id] = helper(as[gid], nbits, pos);

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int index = buffer[local_id] * get_num_groups(0) + group_id;
    unsigned int index_lower;
    if (index > 0) {
        index_lower = counters[index - 1];
    } else {
        index_lower = 0;
    }

    unsigned int shift = 0;

    for (int i = 0; i < local_id; ++i) {
        if (buffer[i] == buffer[local_id]) {
            ++shift;
        }
    }
    bs[index_lower + shift] = as[gid];
}