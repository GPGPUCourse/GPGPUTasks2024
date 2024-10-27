#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

__kernel void pref_sum_naive(__global unsigned int *prev, __global unsigned int *next, int offset, int n) {
    int gid = get_global_id(0);

    int ind_to_add = gid + offset;
    if (ind_to_add >= n) return;

    next[ind_to_add] += prev[gid];
}

__kernel void pref_sum_efficient(__global unsigned int *ps, int offset, int n, int part) {
    int gid = get_global_id(0);

    int ind = part == 0 ?
              (gid + 1) * (offset << 1) - 1 - offset : // first part of algo
              (gid + 1) * (offset << 1) - 1; // second part of algo

    int ind_to_add = ind + offset;
    if (ind_to_add >= n) return;

//    printf("Indices are: ind=%d, ind_to_add=%d\n", ind, ind_to_add);

    ps[ind_to_add] += ps[ind];
}
