#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(bool and_eq, __global const int *arr, unsigned int left, unsigned int right, int val) {
    while (right > left + 1) {
        unsigned int mid = (right + left) / 2;
        if ((and_eq && arr[mid] >= val) || arr[mid] > val) {
            right = mid;
        } else {
            left = mid;
    }
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size) {
    int gid = get_global_id(0);
    int local_idx = gid % block_size;
    int base_idx = (gid / block_size) * block_size * 2;

    int as_index = base_idx + local_idx;
    int left_limit = base_idx - 1 + block_size;
    int right_limit = base_idx + block_size + block_size;

    int current_val = as[as_index];
    unsigned int left_insert_pos = binary_search(1, as, left_limit, right_limit, current_val);
    bs[local_idx + left_insert_pos - block_size] = current_val;

    as_index += block_size;
    current_val = as[as_index];
    left_limit -= block_size;
    right_limit -= block_size;

    unsigned int right_insert_pos = binary_search(0, as, left_limit, right_limit, current_val);
    bs[local_idx + right_insert_pos] = current_val;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
