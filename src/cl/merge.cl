#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int bin_search_lt(__global const int *as, unsigned int n, unsigned int block_begin_idx, unsigned int block_size, int val)
{
    unsigned int l = block_begin_idx;
    unsigned int r = block_begin_idx + block_size + 1;
    l = l >= n + 1 ? n + 1 : l;
    r = r >= n + 1 ? n + 1 : r;

    while (r - l != 1) {
        unsigned int m = (r - l) / 2 + l - 1;
        if (as[m] < val) {
            l = m + 1;
        } else {
            r = m + 1;
        }
    }

    return r - 1;
}

unsigned int bin_search_le(__global const int *as, unsigned int n, unsigned int block_begin_idx, unsigned int block_size, int val)
{
    unsigned int l = block_begin_idx;
    unsigned int r = block_begin_idx + block_size + 1;
    l = l >= n + 1 ? n + 1 : l;
    r = r >= n + 1 ? n + 1 : r;

    while (r - l != 1) {
        unsigned int m = (r - l) / 2 + l - 1;
        if (as[m] <= val) {
            l = m + 1;
        } else {
            r = m + 1;
        }
    }

    return r - 1;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int n)
{
    unsigned int idx = get_global_id(0);

    if (idx >= n) {
        return;
    }

    unsigned int block_idx = idx / block_size;
    unsigned int new_idx;
    if (block_idx % 2 == 0) {
        unsigned int other_block_begin_idx = idx - idx % block_size + block_size;
        new_idx = idx + bin_search_lt(as, n, other_block_begin_idx, block_size, as[idx]) - other_block_begin_idx;
    } else {
        unsigned int other_block_begin_idx = idx - idx % block_size - block_size;
        new_idx = idx - block_size + bin_search_le(as, n, other_block_begin_idx, block_size, as[idx]) - other_block_begin_idx;
    }

    bs[new_idx] = as[idx];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
