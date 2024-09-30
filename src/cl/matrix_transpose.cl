#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float *as, __global float *as_t, unsigned int M, unsigned int K)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    as_t[M * i + j] = as[K * j + i];
}

#define TILE_SIZE 32
__kernel void matrix_transpose_local_bad_banks(__global float *as, __global float *as_t, unsigned int M, unsigned int K)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    tile[local_i][local_j] = as[K * j + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[M * i + j] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(__global float *as, __global float *as_t, unsigned int M, unsigned int K)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    __local float tile[TILE_SIZE + 1][TILE_SIZE];

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    tile[local_i][local_j] = as[K * j + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[M * i + j] = tile[local_i][local_j];
}
