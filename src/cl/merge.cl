#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

int lower_bound(__global const int* as, int size, int value) {
    int l = -1;
    int r = size;
    while (r - l > 1) {
        int m = (l + r) / 2;
        if (as[m] >= value) {
            r = m;
        } else {
            l = m;
        }
    }
    return r;
}

int upper_bound(__global const int* as, int size, int value) {
    int l = -1;
    int r = size;
    while (r - l > 1) {
        int m = (l + r) / 2;
        if (as[m] > value) {
            r = m;
        } else {
            l = m;
        }
    }
    return r;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);
    int pair_start = gid - (gid % (2 * block_size));

    int block_id = gid / block_size;
    int block_idx = gid - block_id * block_size;

    int adjacent_idx;
    if (block_id % 2 == 0) {
        adjacent_idx = lower_bound(as + (block_id + 1) * block_size, block_size, as[gid]);
    } else {
        adjacent_idx = upper_bound(as + (block_id - 1) * block_size, block_size, as[gid]);
    }

    int idx = pair_start + block_idx + adjacent_idx;

    bs[idx] = as[gid];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
