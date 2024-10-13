#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int block_index = gid / block_size;
    unsigned int index_in_block = gid % block_size;

    unsigned int pair_block_index;
    unsigned int left_block_start;
    bool is_left_block = block_index % 2 == 0;
    if (is_left_block) {
        pair_block_index = block_index + 1;
        left_block_start = block_index * block_size;
    } else {
        pair_block_index = block_index - 1;
        left_block_start = (block_index - 1) * block_size;
    }

    unsigned int pair_block_start = pair_block_index * block_size;
    unsigned int left = 0;
    unsigned int right = block_size;
    unsigned int middle;

    int value = as[gid];

    while (left < right) {
        middle = (left + right) / 2;
        if (as[pair_block_start + middle] < value ||
            (is_left_block && as[pair_block_start + middle] == value)) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }

    bs[left_block_start + right + index_in_block] = value;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
