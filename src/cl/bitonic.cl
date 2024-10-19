void swap(__global int* a, __global int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__kernel void bitonic(__global int* as, unsigned int block_size, unsigned int subblock_size)
{
    int gid = get_global_id(0);
    int block_idx = gid % (2 * block_size);
    int subblock_idx = gid % (2 * subblock_size);

    if (block_idx < block_size && subblock_idx < subblock_size) {
        if (as[gid] > as[gid + subblock_size]) {
            swap(&as[gid], &as[gid + subblock_size]);
        }
    } else if (block_idx >= block_size && subblock_idx >= subblock_size) {
        if (as[gid] > as[gid - subblock_size]) {
            swap(&as[gid], &as[gid - subblock_size]);
        }
    }
}
