#define WORKGROUP_SIZE (128)
#define NBITS (4)
#define TRANSPOSE_GROUP_SIZE (16)

__kernel void write_zeros(__global unsigned int *as)
{
    as[get_global_id(0)] = 0;
}

__kernel void radix_count(__global unsigned int *as, __global unsigned int *counters, const unsigned int bit_shift)
{
    const unsigned int gid = get_global_id(0);

    unsigned int bit = (as[gid] >> bit_shift) & ((1 << NBITS) - 1);
    atomic_inc(&counters[get_group_id(0) * (1 << NBITS) + bit]);
}

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *counters_transposed, const unsigned int bit_shift)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int bits[WORKGROUP_SIZE];
    bits[lid] = (as[gid] >> bit_shift) & ((1 << NBITS) - 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    // index of C_{wg}^{bit}
    const unsigned int counters_idx = get_num_groups(0) * bits[lid] + get_group_id(0);
    // since our prefix sum is inclusive
    const unsigned int N_lower_than_me = (counters_idx == 0) ? 0 : counters_transposed[counters_idx - 1];

    unsigned int N_equal_to_me = 0;
    for (int i = 0; i < lid; ++i) {
        if (bits[i] == bits[lid]) {
            N_equal_to_me++;
        }
    }

    bs[N_lower_than_me + N_equal_to_me] = as[gid];
}

// From previous homeworks, only added bounds check for matrix_transpose

__kernel void matrix_transpose(__global unsigned int *as, __global unsigned int *bs, const unsigned int m, const unsigned int k)
{
    __local float block[TRANSPOSE_GROUP_SIZE][TRANSPOSE_GROUP_SIZE + 1];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);

    if (j < k && i < m) {
        block[local_j][local_i] = as[j * m + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const unsigned int x = group_j * TRANSPOSE_GROUP_SIZE + local_i;
    const unsigned int y = group_i * TRANSPOSE_GROUP_SIZE + local_j;

    if (y < m && x < k) {
        bs[y * k + x] = block[local_i][local_j];
    }
}

__kernel void prefix_sum_up(__global unsigned int* as, const unsigned int n, const unsigned int step, const unsigned int workSize) {
    const uint gid = get_global_id(0);

    if (gid >= workSize) {
        return;
    }

    as[step * (gid * 2 + 2) - 1] += as[step * (gid * 2 + 1) - 1];
}

__kernel void prefix_sum_down(__global unsigned int* as, const unsigned int n, const unsigned int step, const unsigned int workSize) {
    const uint gid = get_global_id(0);

    if (gid >= workSize) {
        return;
    }

    if (step * (gid + 1) + step / 2 - 1 >= n) {
        return;
    }
    
    as[step * (gid + 1) + step / 2 - 1] += as[step * (gid + 1) - 1];
}