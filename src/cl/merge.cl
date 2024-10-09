#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);
    int bid = gid / block_size;
    int idx = gid % block_size;
    int s = bid * block_size * 2;
    int pos = s + idx;

    int x = as[pos];
    int l = s + block_size - 1;
    int r = s + block_size * 2;

    while (r - l > 1) {
        int m = (r + l) / 2;
        if (as[m] >= x) r = m;
        else l = m;
    }
    bs[r - block_size + idx] = x;

    pos = s + block_size + idx;
    x = as[pos];
    l = s - 1;
    r = s + block_size;

    while (r - l > 1) {
        int m = (r + l) / 2;
        if (as[m] > x) r = m;
        else l = m;
    }
    bs[idx + r] = x;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
