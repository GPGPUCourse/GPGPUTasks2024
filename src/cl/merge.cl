#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 5

int lower_bound(__global const int *begin, __global const int *end, int x) {
    __global const int *start = begin;
    while (begin != end) {
        __global const int *mid = begin + (end - begin) / 2;
        if (*mid < x) begin = mid + 1;
        else end = mid;
    }

    return begin - start;
}

int upper_bound(__global const int *begin, __global const int *end, int x) {
    __global const int *start = begin;
    while (begin != end) {
        __global const int *mid = begin + (end - begin) / 2;
        if (*mid <= x) begin = mid + 1;
        else end = mid;
    }

    return begin - start;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size) {
    const unsigned int block_idx = get_global_id(0) / (block_size * 2);
    unsigned int idx = get_global_id(0) % (block_size * 2);
    __global const int *begin = as + 2 * block_size * block_idx;
    __global const int *mid = begin + block_size;
    __global const int *end = begin + 2 * block_size;
    __global int *out = bs + 2 * block_size * block_idx;

    if (idx >= block_size) {
        idx -= block_size;
        int x = mid[idx];
        __global int *pos = out + upper_bound(begin, mid, x) + idx;
        *pos = x;
    } else {
        int x = begin[idx];
        __global int *pos = out + lower_bound(mid, end, x) + idx;
        *pos = x;
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
