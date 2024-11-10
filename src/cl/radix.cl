#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define SIZE 16

__kernel void matrix_transpose(__global unsigned int* a, __global unsigned int* at, const unsigned int m, const unsigned int k)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local unsigned int buf[SIZE][SIZE + 1];

    unsigned int i1 = j - local_j + local_i;
    unsigned int j1 = i - local_i + local_j;

    if (j < k && i < m)
        buf[local_j][local_i] = a[j * m + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (j1 < m && i1 < k)
        at[j1 * k + i1] = buf[local_i][local_j];
}

__kernel void prefix_sum_up(__global unsigned int* s, unsigned int n, unsigned int p)
{
	unsigned int index = get_global_id(0);
	unsigned int id2 = 2 * (index + 1) * p - 1;
	unsigned int id1 = id2 - p;
	if (id2 < n) {
		s[id2] += s[id1];
  }
}


__kernel void prefix_sum_down(__global unsigned int* s, unsigned int n, unsigned int p)
{
	unsigned int index = get_global_id(0);
	unsigned int id2 = 2 * (index + 1) * p - 1 + p;
	unsigned int id1 = id2 - p;
	if (id2 < n)
		s[id2] += s[id1];
}

__kernel void count(__global unsigned int *ar, __global unsigned int *counters, unsigned int bit_shift, unsigned int n_bits)
{
    unsigned int gid = get_global_id(0);
    unsigned int grid = get_group_id(0);

    int t = (ar[gid] >> bit_shift) & ((1 << n_bits) - 1);
    atomic_inc(&counters[grid * (1 << n_bits) + t]);
}

__kernel void zero(__global unsigned int *as)
{
    int gid = get_global_id(0);
    as[gid] = 0;
}

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *counters, unsigned int bit_shift, unsigned int n_bits, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int grid = get_group_id(0);
    unsigned int lid = get_local_id(0);

    __local unsigned int buf[128];

    buf[lid] = (as[gid] >> bit_shift) & ((1 << n_bits) - 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int ind = buf[lid] * get_num_groups(0) + grid;
    unsigned int lidx;
    if (ind > 0) {
        lidx = buf[ind - 1];
    } else {
        lidx = 0;
    }

    unsigned int sh = 0;

    for (int i = 0; i < lid; ++i) {
        if (buf[i] == buf[lid])
            sh += 1;
    }
    if ((gid >= n) || (sh + lidx >= n)) {
        return;
    }
    bs[sh + lidx] = as[gid];
}
