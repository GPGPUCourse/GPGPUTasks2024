__kernel void bitonic(__global int *as, __global int *bs, int n, int big_block_size, int small_block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int big_block_idx = gid / (big_block_size / 2);
    unsigned int big_block_local_idx = gid % (big_block_size / 2);
    unsigned int small_block_idx = big_block_local_idx / (small_block_size / 2);
    unsigned int small_block_local_idx = big_block_local_idx % (small_block_size / 2);

    unsigned int idx1 = big_block_size * big_block_idx + small_block_size * small_block_idx + small_block_local_idx;
    unsigned int idx2 = big_block_size * big_block_idx + small_block_size * small_block_idx + small_block_local_idx + small_block_size / 2;

    if (idx2 >= n) {
        return;
    }

    if ((big_block_idx % 2 == 0 && as[idx1] > as[idx2]) || (big_block_idx % 2 == 1 && as[idx1] < as[idx2])) {
        bs[idx1] = as[idx2];
        bs[idx2] = as[idx1];
    } else {
        bs[idx1] = as[idx1];
        bs[idx2] = as[idx2];
    }
}
