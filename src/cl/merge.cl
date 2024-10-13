#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int n)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= n)
        return;
    unsigned int block_id = global_id / block_size;
    unsigned int block_pos = global_id % block_size;
    int element = as[global_id];
    unsigned int search_block_id = block_id;
    if (block_id % 2 == 0)
        ++search_block_id;
    else
        --search_block_id;
    unsigned int element_index = block_pos;
    if (search_block_id * block_size < n) {
        unsigned int search_block_size = min(block_size, n - block_size * search_block_id);
        unsigned int left_border = search_block_id * block_size;
        unsigned int right_border = left_border + search_block_size;
        while (right_border - left_border > 0) {
            unsigned int middle_index = (left_border + right_border) / 2;
            int middle_element = as[middle_index];
            if (middle_element > element || (middle_element == element && search_block_id > block_id))
                right_border = middle_index;
            else
                left_border = middle_index + 1;
        }
        element_index += (right_border - search_block_id * block_size);
    }
    bs[(block_id / 2) * 2 * block_size + element_index] = element;
}

// #define get_element_first(_i) (as[block_id * block_size * 2 + (_i)])
// #define get_element_second(_i) (as[block_id * block_size * 2 + block_size + (_i)])
// #define is_merge_path_go_down(_i, _j) (get_element_first(_j) > get_element_second(_i))
// __kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size, unsigned int n)
// {
//     unsigned int global_id = get_global_id(0);
//     unsigned int block_id = global_id / (2 * block_size);
//     unsigned int block_pos = global_id % (2 * block_size);
//     if (n < block_id * block_size * 2)
//         return;
//     unsigned int width = min(block_size, n - block_id * block_size * 2);
//     unsigned int height = ((width < block_size) ? 0 : (min(block_size, n - block_id * block_size * 2 - block_size)));
//     if (block_pos >= width)
//         return;
//     unsigned int minimal_i_with_zeros = max(0, (int)width - (int)block_pos);
//     unsigned int maximal_i_with_zeros = min(block_pos, height);
//     while (maximal_i_with_zeros - minimal_i_with_zeros > 0) {
//         unsigned int middle_i_with_zeros = (minimal_i_with_zeros + maximal_i_with_zeros) / 2;
//         if (is_merge_path_go_down(middle_i_with_zeros, block_pos - middle_i_with_zeros))
//             minimal_i_with_zeros = middle_i_with_zeros + 1;
//         else
//             maximal_i_with_zeros = middle_i_with_zeros;
//     }
//     inds[global_id] = minimal_i_with_zeros;
// }

// __kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
// {

// }
