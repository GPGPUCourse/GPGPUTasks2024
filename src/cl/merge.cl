#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);
    unsigned int left = gid * block_size;
    unsigned int mid = min(left + block_size / 2, get_global_size(0));
    unsigned int right = min(left + block_size, get_global_size(0));

    unsigned int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (as[i] <= as[j]) {
            bs[k++] = as[i++];
        } else {
            bs[k++] = as[j++];
        }
    }
    while (i < mid) {
        bs[k++] = as[i++];
    }
    while (j < right) {
        bs[k++] = as[j++];
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{
    int gid = get_global_id(0);
    unsigned int left = gid * block_size;
    unsigned int mid = min(left + block_size / 2, get_global_size(0));
    unsigned int right = min(left + block_size, get_global_size(0));

    inds[2 * gid] = left;
    inds[2 * gid + 1] = right;
}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);
    
    unsigned int left = inds[2 * gid];
    unsigned int right = inds[2 * gid + 1];
    unsigned int mid = min(left + block_size / 2, get_global_size(0));

    unsigned int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (as[i] <= as[j]) {
            bs[k++] = as[i++];
        } else {
            bs[k++] = as[j++];
        }
    }
    while (i < mid) {
        bs[k++] = as[i++];
    }
    while (j < right) {
        bs[k++] = as[j++];
    }
}
