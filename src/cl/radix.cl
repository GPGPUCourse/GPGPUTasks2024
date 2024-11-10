

#define TILE_SIZE 16


__kernel void matrix_transpose_local_good_banks(__global unsigned int* a, __global unsigned int* at, const unsigned int m, const unsigned int k)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int new_i = j - local_j + local_i;
    unsigned int new_j = i - local_i + local_j;

    __local unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    unsigned int tile_idx = (local_i + local_j) % TILE_SIZE;
    if (j < k && i < m)
        tile[local_j][tile_idx] = a[j * m + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (new_j < m && new_i < k)
        at[new_j * k + new_i] = tile[local_i][tile_idx];
}


__kernel void prefix_sum_efficient_first(__global unsigned int* as, unsigned int n, unsigned int step)
{
	unsigned int idx = get_global_id(0);
	unsigned int second = (idx + 1) * step * 2 - 1;
	unsigned int first = second - step;
	if (second < n)
		as[second] = as[first] + as[second];
}


__kernel void prefix_sum_efficient_second(__global unsigned int* as, unsigned int n, unsigned int step)
{
	unsigned int idx = get_global_id(0);
	unsigned int second = (idx + 1) * step * 2 - 1 + step;
	unsigned int first = second - step;
	if (second < n)
		as[second] = as[first] + as[second];
}


__kernel void set_zeros(__global unsigned int *as)
{
    unsigned int idx = get_global_id(0);
    as[idx] = 0;
}


__kernel void count(__global unsigned int *as, __global unsigned int *counts, unsigned int bit_shift, unsigned int n_bits)
{
    unsigned int idx = get_global_id(0);
    unsigned int gidx = get_group_id(0);

    unsigned int bit = (as[idx] >> bit_shift) & ((1 << n_bits) - 1);
    atomic_inc(&counts[gidx * (1 << n_bits) + bit]);
}


#define WORKGROUP_SIZE 128

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs,
                         __global unsigned int *counters, unsigned int bit_shift, unsigned int n_bits, unsigned int n)
{
    unsigned int idx = get_global_id(0);
    unsigned int gidx = get_group_id(0);
    unsigned int lidx = get_local_id(0);

    __local unsigned int values[WORKGROUP_SIZE];

    values[lidx] = (as[idx] >> bit_shift) & ((1 << n_bits) - 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int counter = values[lidx] * get_num_groups(0) + gidx;
    unsigned int previous_workgroups = (counter > 0) ? counters[counter - 1] : 0;

    unsigned int curr_workgroup = 0;

    for (int i = 0; i < lidx; ++i)
    {
        if (values[i] == values[lidx])
            curr_workgroup++;
    }
    if ((idx < n) && (previous_workgroups + curr_workgroup < n))
        bs[previous_workgroups + curr_workgroup] = as[idx];
}
