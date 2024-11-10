#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

unsigned int get_value_of_bits(unsigned int val, unsigned int bit_shift, unsigned int nbits) {
    return (val << (32 - (bit_shift + nbits))) >> (32 - nbits);
}

__kernel void reset_to_zeros(__global unsigned *as) {
    as[get_global_id(0)] = 0;
}

__kernel void count(__global unsigned int *as, __global unsigned int *cs,
                    int bit_shift, int nbits) {
    int gid = get_global_id(0);
    int wid = get_group_id(0);

    unsigned d = wid * (1 << nbits) + get_value_of_bits(as[gid], bit_shift, nbits);

    atomic_add(&cs[d], 1);
}

#define TILE_SIZE 16
__kernel void matrix_transpose(__global unsigned int *as,
                               __global unsigned int *as_t, unsigned int w, unsigned int h) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    as_t[gi * h + gj] = as[gj * w + gi];
//    __local unsigned tile[TILE_SIZE][TILE_SIZE + 1];
//    int li = get_local_id(0);
//    int lj = get_local_id(1);
//
//    tile[li][lj] = as[gj * w + gi];
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    int gj_t = gi - (li - lj);
//    int gi_t = gj - (lj - li);
//    as_t[gj_t * w + gi_t] = tile[lj][li];
}

__kernel void pref_sum(__global unsigned int *ps, int offset, int n, int part) {
    int gid = get_global_id(0);

    int ind = part == 0 ?
              (gid + 1) * (offset << 1) - 1 - offset : // first part of algo
              (gid + 1) * (offset << 1) - 1; // second part of algo

    int ind_to_add = ind + offset;
    if (ind_to_add >= n) return;

    ps[ind_to_add] += ps[ind];
}

__kernel void radix_sort(
        __global unsigned int *as, __global unsigned int *res,
        __global unsigned int *ps,
        int bit_shift, int nbits
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);
    int ngroups = get_num_groups(0);

    unsigned int val = get_value_of_bits(as[gid], bit_shift, nbits);
    unsigned int pos = ps[val * ngroups + wid];
    for (int i = 0; i + lid < get_local_size(0); i++) {
        if (val == get_value_of_bits(as[gid + i], bit_shift, nbits)) pos--;
    }

//    printf("pos_in_wg=%d, global_pos=%d, val=%d, num_groups=%d, wid=%d\n", pos_in_wg, global_pos, val, ngroups, wid);
    res[pos] = as[gid];
}
