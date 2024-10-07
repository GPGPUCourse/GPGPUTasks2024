#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int bin_search_lt(__global const int *as, unsigned int block_begin_idx, unsigned int block_size, int val)
{
    unsigned int l = block_begin_idx;
    unsigned int r = block_begin_idx + block_size;

    while (r - l != 1) {
        unsigned int m = (r - l) / 2 + l;
        if (as[m] < val) {
            l = m;
        } else {
            r = m;
        }
    }

    return l;
}

unsigned int bin_search_le(__global const int *as, unsigned int block_begin_idx, unsigned int block_size, int val)
{
    unsigned int l = block_begin_idx;
    unsigned int r = block_begin_idx + block_size;

    while (r - l != 1) {
        unsigned int m = (r - l) / 2 + l;
        if (as[m] <= val) {
            l = m;
        } else {
            r = m;
        }
    }

    return l;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int n)
{
    unsigned int idx = get_global_id(0);

    if (idx >= n) {
        return;
    }

    unsigned int new_idx = block_idx % 2 == 0
        ? idx + bin_search_lt(as, idx % block_size + block_size, block_size, as[idx])
        : idx - block_size + bin_search_le(as, idx % block_size - block_size, block_size, as[idx]);
    bs[new_idx] = as[gid];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
