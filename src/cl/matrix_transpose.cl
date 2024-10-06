#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

__kernel void matrix_transpose_naive(
        __global float *as,
        __global float *as_t,
        const unsigned int M,
        const unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= M || j >= K)
        return;

    as_t[j * M + i] = as[i * K + j];
}

#define TILE_SIZE 16
__kernel void matrix_transpose_local_bad_banks(
        __global float *as,
        __global float *as_t,
        const unsigned int M,
        const unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    unsigned int group_i = get_group_id(0);
    unsigned int group_j = get_group_id(1);

    unsigned int i_new = group_i * TILE_SIZE + local_j;
    unsigned int j_new = group_j * TILE_SIZE + local_i;

    tile[local_j][local_i] = as[j * M + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[i_new * K + j_new] = tile[local_i][local_j];
}

#define TILE_SIZE 16
__kernel void matrix_transpose_local_good_banks(
        __global float *as,
        __global float *as_t,
        const unsigned int M,
        const unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE * (TILE_SIZE + 1)];

    unsigned int group_i = get_group_id(0);
    unsigned int group_j = get_group_id(1);

    unsigned int i_new = group_i * TILE_SIZE + local_j;
    unsigned int j_new = group_j * TILE_SIZE + local_i;

    tile[local_j * (TILE_SIZE + 1) + local_i] = as[j * M + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[i_new * K + j_new] = tile[local_i * (TILE_SIZE + 1) + local_j];
}
