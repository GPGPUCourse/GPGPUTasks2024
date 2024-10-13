#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int lower_bound(__global const int *as, const int value, unsigned int left, unsigned int right) {
    while (right - left > 1) {
        unsigned int mid = (right + left) / 2;
        if (as[mid] >= value) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return right;
}

unsigned int upper_bound(__global const int *as, const int value, unsigned int left, unsigned int right) {
    while (right - left > 1) {
        unsigned int mid = (right + left) / 2;

        if (as[mid] > value) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return right;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    
    unsigned int block_id = gid / block_size;
    unsigned int element_block_index = gid % block_size;

    unsigned int merge_block_start = block_id * block_size * 2;

    unsigned int insert_block_index = lower_bound(
        as, 
        as[merge_block_start + element_block_index], 
        merge_block_start + block_size - 1, 
        merge_block_start + block_size * 2
    );

    bs[insert_block_index - block_size + element_block_index] = as[merge_block_start + element_block_index];

    insert_block_index = upper_bound(
        as, 
        as[merge_block_start + block_size + element_block_index], 
        merge_block_start - 1, merge_block_start + block_size
    );
    
    bs[element_block_index + insert_block_index] = as[merge_block_start + block_size + element_block_index];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
