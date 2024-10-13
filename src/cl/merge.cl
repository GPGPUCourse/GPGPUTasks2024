#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int lower_bound(__global const int *as, unsigned int first, unsigned int last, int value) {
    while (first < last) {
        unsigned int middle = (first + last) / 2;
        if (as[middle] >= value) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

unsigned int upper_bound(__global const int *as, unsigned int first, unsigned int last, int value) {
    while (first < last) {
        unsigned int middle = (first + last) / 2;
        if (as[middle] > value) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return last;
}


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    const int val = as[gid];

    if (gid > n) {
        return;
    }

    unsigned int bid = gid / (block_size * 2);
    unsigned int eid = gid % (block_size * 2);

    unsigned int block_ind = bid * (block_size * 2);

    if (eid < block_size) {
        unsigned int first = block_ind + block_size;
        unsigned int last = first + block_size;
        unsigned int lb_ind = lower_bound(as, first, last, val) - first;
        bs[block_ind + eid % block_size + lb_ind] = val;
    } else {
        unsigned int first = block_ind;
        unsigned int last = first + block_size;
        unsigned int ub_ind = upper_bound(as, first, last, val) - first;
        bs[block_ind + eid % block_size + ub_ind] = val;
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
