#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int upper_bound(__global const int *array, unsigned int len, int val)
{
    unsigned int l = 0;
    unsigned int r = len;

    while (l < r) {
        unsigned int mid = (l + r) / 2;
        int cur = array[mid];
        if (cur <= val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

unsigned int lower_bound(__global const int *array, unsigned int len, int val)
{
    unsigned int l = 0;
    unsigned int r = len;

    while (l < r) {
        unsigned int mid = (l + r) / 2;
        int cur = array[mid];
        if (cur < val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int i = get_global_id(0);
    int val = as[i];

    const unsigned int cur_index = i % block_size;
    const unsigned int cur_block_begin = i - cur_index;
    __global const int *cur = as + cur_block_begin;
    unsigned int other_index = 0;

    if (cur_block_begin % (block_size * 2) == 0) {
        __global const int *other = cur + block_size;
        other_index = lower_bound(other, block_size, val);
    } else {
        __global const int *other = cur - block_size;
        other_index = upper_bound(other, block_size, val);
    }

    bs[i - i % (2 * block_size) + cur_index + other_index] = val;
    return;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
