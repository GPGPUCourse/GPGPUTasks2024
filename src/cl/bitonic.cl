#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6
__kernel void bitonic(__global int *as, unsigned int block_size_log, unsigned int max_block_size_log) {
    int global_id = get_global_id(0);

    int half_block_size = (1 << (block_size_log)) >> 1;
    int block_id = global_id >> (block_size_log - 1);
    int max_block_id = global_id >> (max_block_size_log - 1);
    int is_increasing = (max_block_id & 1) ^ 1;

    int arr_id = global_id + (block_id << (block_size_log - 1));

    int left_elem = as[arr_id];
    int right_elem = as[arr_id + half_block_size];

    if ((left_elem < right_elem) != is_increasing) {
        as[arr_id] = right_elem;
        as[arr_id + half_block_size] = left_elem;
    }
}