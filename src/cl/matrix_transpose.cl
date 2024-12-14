#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float const * const as,
                                     __global float * const as_t,
                                     unsigned int const M,
                                     unsigned int const K)
{
    unsigned int const i = get_global_id(0);
    unsigned int const j = get_global_id(1);

    as_t[i * M + j] = as[j * K + i];
}

#define TILE_SIZE 32
__kernel void matrix_transpose_local_bad_banks(__global float const * const as,
                                               __global float * const as_t,
                                               unsigned int const M,
                                               unsigned int const K)
{
    unsigned int const i = get_global_id(0);
    unsigned int const j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    unsigned int const local_i = get_local_id(0);
    unsigned int const local_j = get_local_id(1);

    tile[local_j][local_i] = as[j * K + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[i * M + j] = tile[local_j][local_i];
}

__kernel void matrix_transpose_local_good_banks(__global float const * const as,
                                               __global float * const as_t,
                                               unsigned int const M,
                                               unsigned int const K)
{
    unsigned int const i = get_global_id(0);
    unsigned int const j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    unsigned int const local_i = get_local_id(0);
    unsigned int const local_j = get_local_id(1);

    tile[local_j][local_i] = as[j * K + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[i * M + j] = tile[local_j][local_i];
}
