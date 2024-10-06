#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(__global const float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    at[j * k + i] = a[i * m + j];
}

__kernel void matrix_transpose_local_bad_banks(__global const float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i_res = get_group_id(0) * TILE_SIZE + local_j;
    int j_res = get_group_id(1) * TILE_SIZE + local_i;
    at[i_res * m + j_res] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(__global const float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i_res = get_group_id(0) * TILE_SIZE + local_j;
    int j_res = get_group_id(1) * TILE_SIZE + local_i;
    at[i_res * m + j_res] = tile[local_i][local_j];
}
