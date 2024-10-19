#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

int binary_search(__global const int *bs, int i, int j, int key, bool lower_bound) {
    int l = -1;
    int r = j - i;
    while (l < r - 1) {
        int m = (l + r) / 2;
        if (bs[i + m] < key || (lower_bound & (bs[i + m] == key))) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size) {
    int global_id = get_global_id(0);
    int elem = as[global_id];
    int block_id = global_id / block_size;
    bool is_left_block = block_id % 2 == 0;

    int block_start = block_id * block_size;
    int paired_block_start = block_id * block_size + (is_left_block ? block_size : -block_size);
    int block_pair_start = block_start + (is_left_block ? 0 : -block_size);

    int offset_within_block = global_id - block_start;
    int offset_within_paired_block = binary_search(as, paired_block_start, paired_block_start + block_size, elem, is_left_block);

    bs[block_pair_start + offset_within_block + offset_within_paired_block] = elem;
}