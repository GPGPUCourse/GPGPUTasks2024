#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
#define BITS 4

__kernel void matrix_transpose(__global   const float* a,
                               __global   float* at,
                               /* rows */ unsigned int m,
                               /* cols */ unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float buf[TILE_SIZE + 1][TILE_SIZE];

    buf[local_i][local_j] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int t_y = i + (local_j - local_i);
    int t_x = j + (local_i - local_j);

    at[t_y * m + t_x] = buf[local_j][local_i];
}

__kernel void prefix_sum(__global const unsigned int* array, __global unsigned int* prefix_sum, unsigned int offset)
{
    int gid = get_global_id(0);

    if (gid < offset) {
        prefix_sum[gid] = array[gid];
        return;
    }

    prefix_sum[gid] = array[gid - offset] + array[gid];
}

__kernel void prefix_sum_up(__global unsigned int* prefix_sum, unsigned int n, unsigned int power)
{
    int gid = get_global_id(0);

    int idx = 2 * (gid + 1) * power - 1;
    if (idx < 0 || idx >= n) {
        return;
    }

    prefix_sum[idx] += prefix_sum[idx - power];
}

__kernel void prefix_sum_down(__global unsigned int* prefix_sum, unsigned int n, unsigned int power)
{
    int gid = get_global_id(0);
    int idx = power + 2 * (gid + 1) * power - 1;

    if (idx < 0 || idx >= n) {
        return;
    }

    prefix_sum[idx] += prefix_sum[idx - power];
}

__kernel void count_local_entries(__global const unsigned int* array,
                                  __global unsigned int* result,
                                  unsigned int offset)
{
    int gid  = get_global_id(0);
    int lid = get_local_id(0);
    int wgid = get_group_id(0);
    int wgsz = get_local_size(0);
    unsigned int mask = ((1 << BITS) - 1) << offset;

    __local unsigned int counter[1 << BITS];

    if (lid < (1 << BITS)) {
        counter[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int index = (array[gid] & (mask)) >> offset;
    atomic_add(&counter[index], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < (1 << BITS)) {
        result[wgid * (1 << BITS) + lid] = counter[lid];
    }
}

__kernel void radix_sort(__global const unsigned int* array,
                         __global unsigned int* result,
                         __global const unsigned int* pref_sum_entries,
                         unsigned int offset)
{
    int gid  = get_global_id(0);
    int lid = get_local_id(0);
    int wgid = get_group_id(0);
    int wgsz = get_local_size(0);
    int wgcnt = get_num_groups(0);
    unsigned int mask = ((1 << BITS) - 1) << offset;

    unsigned int current = array[gid];

    unsigned int numWithOffset = (current & mask) >> offset;
    unsigned int equalNums = 0;
    for (unsigned int i = 0; i < lid; ++i) {
        unsigned int n = (array[wgsz * wgid + i] & mask) >> offset;
        if (numWithOffset == n) {
            ++equalNums;
        }
    }

    unsigned int index = equalNums;

    if (numWithOffset != 0 || wgid != 0) {
        index += pref_sum_entries[wgcnt * numWithOffset + wgid - 1];
    }

    result[index] = current;
}
