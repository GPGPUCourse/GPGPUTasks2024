#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    const size_t gid = get_global_id(0);
    const size_t idx_in_block = gid % block_size;
    const size_t block_base = gid - idx_in_block;

    if (idx_in_block < block_size / 2) {
        int l = block_base + block_size / 2 - 1;
        int r = block_base + block_size;
        while (l + 1 < r) {
            int m = l + (r - l) / 2;
            if (as[m] < as[gid]) {
                l = m;
            } else {
                r = m;
            }
        }
        bs[idx_in_block + l - block_size / 2 + 1] = as[gid];
    } else {
        int l = block_base - 1;
        int r = block_base + block_size / 2;
        while (l + 1 < r) {
            int m = l + (r - l) / 2;
            if (as[m] <= as[gid]) {
                l = m;
            } else {
                r = m;
            }
        }
        bs[idx_in_block - block_size / 2 + l + 1] = as[gid];
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
