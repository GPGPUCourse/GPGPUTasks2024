#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(bool equal, __global const int *as, int left, int right, int value) {
    while (right > left + 1) {
        int m = (right + left) / 2;
        if ((equal && as[m] >= value) || as[m] > value) {
            right = m;
        } else {
            left = m;
        }
    }
    return left + 1;
}


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{

    int gid = get_global_id(0);
    int index = gid % block_size;
    int start_index = (gid / block_size) * block_size * 2;

    int value_idx = start_index + index;
    int left = start_index - 1 + block_size;
    int right = start_index + block_size + block_size;

    int value = as[start_index + index];
    unsigned int diff_idx_l = binary_search(1, as, left, right, value);
    bs[index + diff_idx_l - block_size] = value;

    value_idx += block_size;
    left -= block_size;
    right -= block_size;

    value = as[value_idx];
    unsigned int diff_idx_r = binary_search(0, as, left, right, value);
    bs[index + diff_idx_r] = value;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
