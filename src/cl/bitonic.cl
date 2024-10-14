__kernel void bitonic(__global int *as, __global int *bs, int n, int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int block_idx = gid / (block_size / 2);
    unsigned int inblock_idx = gid % (block_size / 2);
    unsigned int idx1 = block_size * block_idx;
    unsigned int idx2 = block_size * block_idx + block_size / 2;

    if (idx2 >= n) {
        return;
    }

    if (block_idx % 2 == 0) {
        if (as[idx1] > as[idx2]) {
            bs[idx1] = as[idx2];
            bs[idx2] = as[idx1];
        }
    } else {
        if (as[idx2] > as[idx1]) {
            bs[idx1] = as[idx2];
            bs[idx2] = as[idx1];
        }
    }
}
