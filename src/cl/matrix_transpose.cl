#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

__kernel void matrix_transpose_naive(
        __global const float *a,
        __global float *at,
        const unsigned int M,
        const unsigned int K
) {
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);
    if (i < M && j < K)
        at[j * M + i] = a[i * K + j];
}

#define TILE_SIZE 16
__kernel void matrix_transpose_local_bad_banks(
        __global const float *a,
        __global float *at,
        const unsigned int M,
        const unsigned int K
) {
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    const unsigned int jj = get_local_id(0);
    const unsigned int ii = get_local_id(1);

    __local float cache[TILE_SIZE][TILE_SIZE];

    if (i < M && j < K)
        cache[ii][jj] = a[i * K + j];
    else
        cache[ii][jj] = 0;
// почему то оператор ниже не работает, если выключить верхний if :(
//    cache[ii][jj] = (i < M && j < K) ? a[i * K + j] : 0;
    float value = cache[ii][jj];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (jj < ii){
        cache[ii][jj] = cache[jj][ii];
        cache[jj][ii] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int target_j = get_group_id(0) * TILE_SIZE + ii;
    const unsigned int target_i = get_group_id(1) * TILE_SIZE + jj;
    if (target_i < M && target_j < K)
        at[target_j * M + target_i] = cache[ii][jj];
}

__kernel void matrix_transpose_local_good_banks(
        __global const float *a,
        __global float *at,
        const unsigned int M,
        const unsigned int K

) {
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    const unsigned int jj = get_local_id(0);
    const unsigned int ii = get_local_id(1);

    __local float cache[TILE_SIZE][TILE_SIZE];

    float value = (i < M && j < K) ? a[i * K + j] : 0;
    cache[jj][(ii + jj) % TILE_SIZE] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int target_j = get_group_id(0) * TILE_SIZE + ii;
    const unsigned int target_i = get_group_id(1) * TILE_SIZE + jj;
    if (target_i < M && target_j < K)
        at[target_j * M + target_i] = cache[ii][(ii + jj) % TILE_SIZE];
}
