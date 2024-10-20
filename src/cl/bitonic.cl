#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6
__kernel void bitonic(__global int *as, unsigned int block_size, unsigned int max_block_size) {
    int global_id = get_global_id(0);

    int half_block_size = block_size / 2;

    int block_id = global_id / half_block_size;
    int max_block_id = global_id / (max_block_size / 2);
    int is_increasing = (max_block_id % 2 == 0);

    int arr_id = global_id + block_id * half_block_size;

//    printf("comparing lhs_id=%d rhs_id=%d lhs=%d rhs=%d is_increasing=%d block_size=%d\n",
//           arr_id, arr_id + half_block_size, as[arr_id], as[arr_id + half_block_size], is_increasing, block_size);
    if ((is_increasing && as[arr_id] > as[arr_id + half_block_size])
        || (!is_increasing && as[arr_id] < as[arr_id + half_block_size])) {
//        printf("swap\n");
        unsigned int tmp = as[arr_id + half_block_size];
        as[arr_id + half_block_size] = as[arr_id];
        as[arr_id] = tmp;
    }
}