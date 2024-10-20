__kernel void bitonic(__global int *as, unsigned int block_half_size, unsigned int sub_block_half_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int block_index = block_index = gid / block_half_size;
    bool is_growing = block_index % 2 == 0;
    unsigned int idx = gid / sub_block_half_size * (sub_block_half_size * 2) + (gid % sub_block_half_size);

    unsigned int pair_index = idx + sub_block_half_size;
    if (is_growing && as[idx] > as[pair_index] ||
        !is_growing && as[idx] < as[pair_index]
    ) {
        int tmp = as[idx];
        as[idx] = as[pair_index];
        as[pair_index] = tmp;
    }
}
