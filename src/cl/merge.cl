#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(__global const int* as, int l, int r, int val, bool canBeEqual = false) {
    while (r > l + 1) {
        int m = (r + l) / 2;

        if (as[m] > val || (canBeEqual && as[m] >= val)) {
            r = m;
        } else {
            l = m;
        }
    }

    return l + 1;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);
    int begin = (gid / block_size) * block_size * 2;

    int i = gid % block_size;
    int i_val = begin + i;

    int l = begin + block_size - 1;
    int r = begin + 2 * block_size;

    // search in the right half
    int val = as[begin + i];
    unsigned int count = binary_search(as, l, r, val, true);
    bs[i + count - block_size] = val;

    // search in the left half
    i_val += block_size;
    val = as[i_val];

    l -= block_size;
    r -= block_size;

    count = binary_search(as, l, r, val);
    bs[i + count] = val;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
