#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

#define TILE_SIZE 8
__kernel void matrix_transpose_naive(
        __global float *a,
        __global float *at,
        unsigned int m,
        unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    at[i * m + j] = a[j * k + i];
}

__kernel void matrix_transpose_local_bad_banks(
        __global float *a,
        __global float *at,
        unsigned int m,
        unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_i = i - local_i;
    int tile_j = j - local_j;

    at[(tile_i + local_j) * k + (tile_j + local_i)] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_bad_banks_from_lecture(
        __global float *a,
        __global float *at,
        unsigned int m,
        unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    // почему "здесь все хорошо с coalesced", если мы все равно загружаем в память по мере изменения j?
    // я понимаю, что разницы нет, если группа помещается на один варп
    // но если нет, то мы будем загружать в память вертикальным прямоугольником (а могли бы горизонтальным)
    at[i * m + j] = tile[local_j][local_i];
}


__kernel void matrix_transpose_local_good_banks(
        __global float *a,
        __global float *at,
        unsigned int m,
        unsigned int k
        ) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_i = i - local_i;
    int tile_j = j - local_j;

    at[(tile_i + local_j) * k + (tile_j + local_i)] = tile[local_i][local_j];
}
