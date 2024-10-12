#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

bool lower(__global const int *as, int gid1, int gid2) {
    return as[gid1] < as[gid2] || as[gid1] == as[gid2] && gid1 < gid2;
}

int lower_bound(__global const int *as, int l, int r, int gid) {
    while (l < r) {
        int mid = (l + r) / 2;
        if (lower(as, mid, gid)) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return r;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);
    int idx = gid % (2 * block_size);
    int start = gid - idx;
    int end = start + 2 * block_size;

    int gid_out;
    if (idx < block_size) {
        gid_out = lower_bound(as, start + block_size, end, gid) + gid - (start + block_size);
    } else {
        gid_out = lower_bound(as, start, start + block_size, gid) + (idx - block_size);
    }

    bs[gid_out] = as[gid];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
