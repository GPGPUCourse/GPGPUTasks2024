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

    as_t[gi * h + gj] = tile[li][lj];
}

__kernel void matrix_transpose_local_good_banks(__global float *as,
                                                __global float *as_t, unsigned int w, unsigned int h) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[lj][li] = as[gj * w + gi];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[gi * h + gj] = tile[lj][li];
}
