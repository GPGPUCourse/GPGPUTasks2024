// kernel.cl

#define TILE_SIZE 16

__kernel void write_zeros(__global unsigned int *counters, unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        counters[gid] = 0;
    }
}

__kernel void count(__global const int *as, 
                    __global int *counters, 
                    unsigned int bit_shift, 
                    unsigned int nbits) {
    unsigned int gid = get_global_id(0);
    unsigned int grid = get_group_id(0);

    unsigned int t = (as[gid] >> bit_shift) & ((1 << nbits) - 1);
    atomic_inc(&counters[grid * (1 << nbits) + t]);
}

__kernel void transpose_counters(__global unsigned int* a, 
                                    __global unsigned int* at, 
                                    const unsigned int k, 
                                    const unsigned int m)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE + 1]; 
    int li = get_local_id(0);
    int lj = get_local_id(1);
    
    tile[lj][li] = a[j * k + i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    at[(i - li + lj) * m + j - lj + li] = tile[li][lj];
}

__kernel void prefix_sum_up(__global int *as, const int offset, const int n) {
    unsigned int gid = get_global_id(0);
    unsigned int right = (gid + 1) * 2 * offset - 1;
    unsigned int left = right - offset;
    if (right < n)
        as[right] = as[left] + as[right];
}

__kernel void prefix_sum_down(__global int *as, const int offset, const int n) {
    unsigned int gid = get_global_id(0);
    unsigned int left = (gid + 1) * 2 * offset - 1;
    unsigned int right = left + offset;
    if (right < n)
        as[right] = as[left] + as[right];
}

__kernel void radix_sort(
    __global const unsigned int* as,
    __global unsigned int* bs, 
    __global unsigned int* counters,
    unsigned int bit_shift,
    unsigned int nbits,
    unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int grid = get_group_id(0);
    unsigned int lid = get_local_id(0);

    __local unsigned int buf[128];

    buf[lid] = (as[gid] >> bit_shift) & ((1 << nbits) - 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int ind = buf[lid] * get_num_groups(0) + grid;
    unsigned int lidx = (ind > 0 && ind < n) ? counters[ind - 1] : 0;

    unsigned int sh = 0;
    for (int i = 0; i < lid; ++i) {
        sh += (buf[i] == buf[lid]);
    }

    unsigned int target_index = sh + lidx;
    if (gid < n && target_index < n) {
        bs[target_index] = as[gid];
    }
}
