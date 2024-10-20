#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

__kernel void matrix_transpose_naive(__global float *as,
                                     __global float *as_t, unsigned int w, unsigned int h) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    as_t[i * h + j] = as[j * w + i];
}

#define TILE_SIZE 32
__kernel void matrix_transpose_local_bad_banks(__global float *as,
                                               __global float *as_t, unsigned int w, unsigned int h) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[li][lj] = as[gj * w + gi];

    barrier(CLK_LOCAL_MEM_FENCE);

    // mirroring over tile main diag
    // (x1,y1) = (li,lj) in old tile
    // (x2,y2) = (lj,li) in new tile
    // (x1-x2, y1-y2) = (li-lj, lj-li) is shift
    // axis swapped in transposed
    // so new coords increase over col, then over row (like initial, so we have coalesced access)
    int gj_t = gi - (li - lj);
    int gi_t = gj - (lj - li);
    as_t[gj_t * w + gi_t] = tile[lj][li];
}

__kernel void matrix_transpose_local_good_banks(__global float *as,
                                                __global float *as_t, unsigned int w, unsigned int h) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);

    // with just 1 extra element in each row we get bank index (li + lj) % 32 for elem on (li, lj) and TILESIZE=32
    // both elems for each thread is in one bank (if 32 banks total)
    // KPACUBO)
    __local float tile[TILE_SIZE + 1][TILE_SIZE];
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[li][lj] = as[gj * w + gi];

    barrier(CLK_LOCAL_MEM_FENCE);

    // mirroring over tile main diag
    // (x1,y1) = (li,lj) in old tile
    // (x2,y2) = (lj,li) in new tile
    // (x1-x2, y1-y2) = (li-lj, lj-li) is shift
    // axis swapped in transposed
    // so new coords increase over col, then over row (like initial, so we have coalesced access)
    int gj_t = gi - (li - lj);
    int gi_t = gj - (lj - li);
    as_t[gj_t * w + gi_t] = tile[lj][li];
}
