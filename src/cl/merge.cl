#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


int bsearch(__global const int *a, unsigned int len, int key, int even) {
    int l = 0;
    int r = len;
    while (l < r) {
        int m = (l + r) / 2;
        if (a[m] < key || (even && a[m] == key)) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int gid = get_global_id(0);

    int pos_in_first = gid % block_size;
    unsigned int first_begin = (gid / block_size) * block_size;
    __global const int* first = as + first_begin;

    int parity = (gid / block_size) % 2;
    __global const int* second =  parity ? first - block_size : first + block_size;
    int pos_in_second = bsearch(second, block_size, as[gid], parity);

    int pair_begin = (gid / (2 * block_size)) * (2 * block_size);
//    printf("Calculated parity %d, pair_begin %d, pos_in_first %d, pos_in_second %d for gid %d, key %d\n", parity, pair_begin, pos_in_first, pos_in_second, gid, as[gid]);

    bs[pair_begin + pos_in_first + pos_in_second] = as[gid];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
