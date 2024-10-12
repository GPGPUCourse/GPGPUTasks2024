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
    const size_t eid = gid % block_size;
    const size_t begin = 2 * bid * block_size;

    const int pivot1 = as[begin + eid];
    size_t index1 = lower_bound(as, pivot1, begin + block_size - 1, begin + 2 * block_size);
    bs[eid + index1 - block_size] = pivot1;

    const int pivot2 = as[begin + eid + block_size];
    size_t index2 = upper_bound(as, pivot2, begin - 1, begin + block_size);
    bs[eid + index2] = pivot2;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
