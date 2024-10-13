#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 5

int bin_search(const int x, const int *as, int l, int r, bool is_strict) {
    while (l <= r) {
        unsigned int m = (l + r) / 2;

        if (as[m] == x) {
            if (is_strict)
                r = m - 1;
            else
                l = m + 1;
        } else if (as[m] < x)
            l = m + 1;
        else
            r = m - 1;
    }
    return l;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int as_size) {
    unsigned int global_i = get_global_id(0);

    if (global_i >= as_size)
        return;

    unsigned int i = global_i % block_size;
    unsigned int first_block_start = global_i / (block_size * 2) * block_size * 2;
    unsigned int second_block_start = min(first_block_start + block_size, as_size);
    unsigned int first_block_end = second_block_start - 1;
    unsigned int second_block_end = min(second_block_start + block_size, as_size) - 1;

    unsigned int j;
    if (global_i % (block_size * 2) == i)
        // i in first block
        j = bin_search(as[first_block_start + i],
                       as,
                       second_block_start,
                       second_block_end,
                       false)
            - second_block_start;
    else
        // i in second block
        j = bin_search(as[second_block_start + i],
                       as,
                       first_block_start,
                       first_block_end,
                       true)
            - first_block_start;
    bs[first_block_start + i + j] = as[global_i];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size) {

}

__kernel void
merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size) {

}
