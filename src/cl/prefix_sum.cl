__kernel void work_efficient_prefix_sum_pass1(
    __global unsigned int* as,
    unsigned int as_size,
    unsigned int block_size
) {
    unsigned int i = get_global_id(0);
    unsigned int last_in_block = (1 + i) * block_size - 1;
    unsigned int mid_in_block = last_in_block - block_size / 2;
    if (last_in_block < as_size) {
        as[last_in_block] = as[mid_in_block] + as[last_in_block];
    }
}

__kernel void work_efficient_prefix_sum_pass2(
    __global unsigned int* as,
    unsigned int as_size,
    unsigned int block_size
) {
    unsigned int i = get_global_id(0);
    unsigned int target_i = (1 + i) * block_size * 2 + block_size - 1;
    if (target_i < as_size) {
        as[target_i] = as[target_i - block_size] + as[target_i];
    }
}
