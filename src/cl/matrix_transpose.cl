#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16
#line 6

__kernel void matrix_transpose_naive( __global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= k)
        return;
    if (j >= m)
        return;
    float x = a[i * k + j];
    at[j * m + i] = x;
}


__kernel void matrix_transpose_local_bad_banks( __global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i >= k)
        return;
    if (j >= m)
        return;

    tile[local_i][local_j] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    at[j * m + i] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks( __global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int indx = local_i * TILE_SIZE + local_j;
    int biased_i= indx / (TILE_SIZE + 1);
    int biased_j = indx % (TILE_SIZE + 1);
    if (i >= k)
        return;
    if (j >= m)
        return;

    tile[biased_i][biased_j] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    at[j * m + i] = tile[biased_i][biased_j];
}
