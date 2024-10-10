#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

size_t lower_bound(__global const int *as, int pivot, size_t from, size_t to)
{
    while (1 < to - from) {
        const size_t mid = (from + to) / 2;
        if (as[mid] < pivot) {
            from = mid;
        } else {
            to = mid;
        }
    }
    return to;
}

size_t upper_bound(__global const int *as, int pivot, size_t from, size_t to)
{
    while (1 < to - from) {
        const size_t mid = (from + to) / 2;
        if (as[mid] <= pivot) {
            from = mid;
        } else {
            to = mid;
        }
    }
    return to;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    const size_t gid = get_global_id(0);
    const size_t bid = gid / block_size;
    const size_t begin = bid * block_size;

    if (bid % 2 == 0) {
        const int pivot = as[gid];
        size_t index = upper_bound(as, pivot, begin + block_size, begin + 2 * block_size);
        bs[gid + index - begin - block_size] = pivot;
    } else {
        const int pivot = as[gid];
        size_t index = lower_bound(as, pivot, begin - block_size, begin);
        bs[gid + index - begin] = pivot;
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
