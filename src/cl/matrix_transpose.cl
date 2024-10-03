#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(__global float *a, __global float *at, const int m, const int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    at[i * m + j] = a[j * k + i];
}

__kernel void matrix_transpose_local_bad_banks(__global float *a, __global float *at, const int M, const int K)
{
    __local float tile[TILE_SIZE * TILE_SIZE];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[li * TILE_SIZE + lj] = a[j * K + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    at[(get_group_id(0) * TILE_SIZE + lj) * K + get_group_id(1) * TILE_SIZE + li] = tile[lj * TILE_SIZE + li];
}

__kernel void matrix_transpose_local_good_banks(__global float *a, __global float *at, const int M, const int K)
{
    __local float tile[TILE_SIZE * (TILE_SIZE + 1)];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[lj * (TILE_SIZE + 1) + li] = a[j * K + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    at[(get_group_id(0) * TILE_SIZE + lj) * K + get_group_id(1) * TILE_SIZE + li] = tile[li * (TILE_SIZE + 1) + lj];
}
