#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(__global const int *arr, int left, int right, const int value, const bool strict)
{
    right--;
    while (left <= right)
    {
        int middle = left + (right - left) / 2;
        int compResult = strict ? (arr[middle] < value) : (arr[middle] <= value);
        if (compResult)
            left = middle + 1;
        else
            right = middle - 1;
    }

    return left;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, const unsigned int n)
{
const unsigned int gidx = get_global_id(0);

    if (gidx >= n)
        return;

    unsigned int blockIndex = gidx / block_size;
    int left;

    if (blockIndex % 2 == 0)
    {
        left = (blockIndex + 1) * block_size;
    }
    else
    {
        left = (blockIndex - 1) * block_size;
    }

    if (left >= n)
    {
        bs[gidx] = as[gidx];
        return;
    }

    unsigned int asIndex = gidx % block_size;
    unsigned int right = left + block_size;
    if (right > n)
        right = n;

    int strict = blockIndex % 2 != 0;
    unsigned int bsIndex = binary_search(as, left, right, as[gidx], strict) - left;
    unsigned int resultingIndex = (blockIndex - (blockIndex % 2)) * block_size + asIndex + bsIndex;

    bs[resultingIndex] = as[gidx];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}