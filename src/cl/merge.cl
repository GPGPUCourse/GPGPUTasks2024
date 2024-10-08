#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int bin_search(int or_eq, __global const int *as, unsigned int block_begin_idx, unsigned int block_end_idx, int val)
{
    unsigned int l = block_begin_idx;
    unsigned int r = block_end_idx + 1;

    while (r - l != 1) {
        unsigned int m = (r - l) / 2 + l - 1;
        if ((or_eq && as[m] <= val) || (!or_eq && as[m] < val)) {
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
        unsigned int other_block_end_idx = other_block_begin_idx + block_size;
        other_block_begin_idx = other_block_begin_idx >= n ? n : other_block_begin_idx;
        other_block_end_idx = other_block_end_idx >= n ? n : other_block_end_idx;
        new_idx = idx + bin_search(0, as, other_block_begin_idx, other_block_end_idx, as[idx]) - other_block_begin_idx;
    } else {
        unsigned int other_block_begin_idx = idx - idx % block_size - block_size;
        unsigned int other_block_end_idx = other_block_begin_idx + block_size;
        other_block_begin_idx = other_block_begin_idx >= n ? n : other_block_begin_idx;
        other_block_end_idx = other_block_end_idx >= n ? n : other_block_end_idx;
        new_idx = idx - block_size + bin_search(1, as, other_block_begin_idx, other_block_end_idx, as[idx]) - other_block_begin_idx;
    }

    bs[new_idx] = as[idx];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
