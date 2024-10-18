__kernel void fill_infinity(__global int *as, unsigned int n, unsigned int n_two_pow)
{
    const unsigned int global_id = get_global_id(0);
    if (n + global_id >= n_two_pow)
        return;
    as[n + global_id] = INT_MAX;
}

#define get_index(_i) as[superblock_index * half_superblock_size * 2 + (half_superblock_size * 2 - 1) * (superblock_index % 2) + (block_index * half_block_size * 2 + (_i)) * (1 - 2 * (superblock_index % 2))]
__kernel void bitonic(__global int *as, unsigned int n, unsigned int half_superblock_size, unsigned int operation_index)
{
    const unsigned int global_id = get_global_id(0);
    if (global_id * 2 >= n)
        return;
    const unsigned int half_block_size = half_superblock_size / operation_index;
    const unsigned int superblock_index = global_id / half_superblock_size;
    const unsigned int index_in_superblock = global_id % half_superblock_size;
    const unsigned int block_index = index_in_superblock / half_block_size;
    const unsigned int index_in_block = index_in_superblock % half_block_size;
    int first = get_index(index_in_block);
    int second = get_index(index_in_block + half_block_size);
    if (second < first) {
        get_index(index_in_block) = second;
        get_index(index_in_block + half_block_size) = first;
    }
}
