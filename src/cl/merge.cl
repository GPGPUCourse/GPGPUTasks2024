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
    return left + 1;
}


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size) {
    int global_id = get_global_id(0);
    int local_index = global_id % block_size;
    int base_index = (global_id / block_size) * block_size * 2;

    int input_index = base_index + local_index;
    int left_bound = base_index - 1 + block_size;
    int right_bound = base_index + block_size + block_size;

    int current_value = as[base_index + local_index];
    unsigned int left_insert_pos = binary_search(1, as, left_bound, right_bound, current_value);
    bs[local_index + left_insert_pos - block_size] = current_value;

    input_index += block_size;
    current_value = as[input_index];
    left_bound -= block_size;
    right_bound -= block_size;

    unsigned int right_insert_pos = binary_search(0, as, left_bound, right_bound, current_value);
    bs[local_index + right_insert_pos] = current_value;
}


__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
