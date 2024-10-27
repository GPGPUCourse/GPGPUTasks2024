#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(__global const int *arr, int left, int right, const int value, const bool strict) {
    right--;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (strict ? (arr[mid] > value) : (arr[mid] >= value))
            right = mid - 1;
        else
            left = mid + 1;
    }

    return left;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int blockIdx = gid / block_size;

    unsigned int left = (blockIdx - 1) * block_size;
    if (blockIdx % 2 == 0)
        left = (blockIdx + 1) * block_size;

    unsigned int asIdx = gid % block_size;
    unsigned int bsIdx = binary_search(as, left, left + block_size, as[gid], blockIdx % 2 == 0) - left;
    unsigned int idx = (blockIdx - (blockIdx % 2)) * block_size + asIdx + bsIdx;

    bs[idx] = as[gid];  
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int blockIdx = gid / block_size;

    unsigned int left = blockIdx * block_size;
    if (blockIdx % 2 != 0)
        left = (blockIdx - 1) * block_size;
    
    unsigned int right = left + block_size;
    unsigned int i = gid - left + 1;

    unsigned int lj = 0;
    unsigned int rj = i - 1;
    if (i > block_size)
        rj = 2 * block_size - i - 1;
    while (lj <= rj) {
        unsigned int j = (lj + rj) / 2;
        if (as[left + j + (0 > i - block_size ? 0 : i - block_size)] < as[right - j + (i < block_size ? i : block_size) - 1])
            lj = j + 1;
        else
            rj = j - 1;
    }
    inds[gid] = lj + (0 > i - block_size ? 0 : i - block_size);
}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int blockIdx = gid / block_size;

    unsigned int left = blockIdx * block_size;
    if (blockIdx % 2 != 0)
        left = (blockIdx - 1) * block_size;
    
    unsigned int right = left + block_size;
    unsigned int i = gid - left;
    
    if (gid == left)
        bs[gid] = (inds[gid] == 0 ? as[right] : as[left]);
    else if (inds[gid] > inds[gid - 1])
        bs[gid] = as[left + (0 > i - block_size + 1 ? 0 : i - block_size + 1) + (inds[gid] - (0 > i - block_size + 1 ? 0 : i - block_size + 1)) - 1];
    else
        bs[gid] = as[right + (i < block_size - 1 ? i : block_size - 1) - (inds[i] - (0 > i - block_size + 1 ? 0 : i - block_size + 1))];
}
