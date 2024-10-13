#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(bool equal, __global const int *arr, unsigned int left, unsigned int right, int val) {
    while (right > left + 1) {
        unsigned int mid = (right + left) / 2;
        if ((equal && arr[mid] >= val) || arr[mid] > val) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return left + 1;
}

__kernel void merge_global(__global const int *input, __global int *output, unsigned int block_size) {
    int gid = get_global_id(0);
    int index = gid % block_size;
    int start_index = (gid / block_size) * block_size * 2;

    int input_index = start_index + index;
    int left = start_index - 1 + block_size;
    int right = start_index + block_size + block_size;

    int current_value = input[input_index];
    unsigned int insert_position_left = binary_search(1, input, left, right, current_value);
    output[index + insert_position_left - block_size] = current_value;

    input_index += block_size;
    left -= block_size;
    right -= block_size;

    current_value = input[input_index];
    unsigned int insert_position_right = binary_search(0, input, left, right, current_value);
    output[index + insert_position_right] = current_value;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
