#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(__global const int *arr, unsigned int left, unsigned int right, int value) {
    while (left < right) {
        unsigned int mid = left + (right - left) / 2;
        if (arr[mid] <= value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__kernel void merge_global(__global const int *input, __global int *output, unsigned int block_size) {
    int gid = get_global_id(0);
    int local_index = gid % block_size;
    int base_index = (gid / block_size) * block_size * 2;

    int input_index = base_index + local_index;
    int left_bound = base_index;
    int right_bound = base_index + 2 * block_size;

    int current_value = input[input_index];
    unsigned int insert_position_left = binary_search(input, base_index + block_size, right_bound, current_value);
    output[local_index + insert_position_left - block_size] = current_value;

    input_index += block_size;
    current_value = input[input_index];
    unsigned int insert_position_right = binary_search(input, left_bound, base_index + block_size, current_value);
    output[local_index + insert_position_right] = current_value;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
