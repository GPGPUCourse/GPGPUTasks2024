__kernel void fill_with_zeros(__global unsigned int *as, unsigned int n, unsigned int n_two_pow)
{
    const unsigned int global_id = get_global_id(0);
    if (n + global_id >= n_two_pow)
        return;
    as[n + global_id] = 0;
}

__kernel void work_efficient_sum(__global unsigned int *as, unsigned int n, unsigned int block_size) {
    const unsigned int global_id = get_global_id(0);
    if (global_id * block_size >= n)
        return;
    const unsigned int left_border = global_id * block_size;
    const unsigned int right_border = left_border + block_size - 1;
    const unsigned int middle_border = left_border + block_size / 2 - 1;
    as[right_border] += as[middle_border];
}

__kernel void refresh(__global unsigned int *as, unsigned int n, unsigned int block_size) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int current_source_position = (global_id + 1) * block_size - 1;
    const unsigned int current_destination_position = current_source_position + block_size / 2;
    if (current_destination_position >= n)
        return;
    as[current_destination_position] += as[current_source_position];
}