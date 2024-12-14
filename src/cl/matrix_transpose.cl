#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float const * const as,
                                     __global float * const as_t,
                                     unsigned int const M,
                                     unsigned int const K)
{
    unsigned int const i = get_global_id(1);
    unsigned int const j = get_global_id(0);

    as_t[i * M + j] = as[j * K + i];
}

#define TILE_SIZE 32
__kernel void matrix_transpose_local_bad_banks(__global float const *const as,
                                               __global float *const as_t,
                                               unsigned int const M,
                                               unsigned int const K)
{
    uint const gi = get_global_id(1);
    uint const gj = get_global_id(0);
    uint const y_group_offset = get_group_id(1) * get_local_size(1);
    uint const x_group_offset = get_group_id(0) * get_local_size(0);
    uint const local_i = get_local_id(1);
    uint const local_j = get_local_id(0);

    __local float tile[TILE_SIZE][TILE_SIZE];

    tile[local_i][local_j] = as[gi * K + gj];

    barrier(CLK_LOCAL_MEM_FENCE);
    uint const t_i = x_group_offset + local_i;
    uint const t_j = y_group_offset + local_j;
    as_t[t_i * M + t_j] = tile[local_j][local_i];
}

__kernel void matrix_transpose_local_good_banks(__global float const *const as,
                                               __global float *const as_t,
                                               unsigned int const M,
                                               unsigned int const K)
{
    uint const gi = get_global_id(1);
    uint const gj = get_global_id(0);
    uint const y_group_offset = get_group_id(1) * get_local_size(1);
    uint const x_group_offset = get_group_id(0) * get_local_size(0);
    uint const local_i = get_local_id(1);
    uint const local_j = get_local_id(0);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    tile[local_i][local_j] = as[gi * K + gj];

    barrier(CLK_LOCAL_MEM_FENCE);
    uint const t_i = x_group_offset + local_i;
    uint const t_j = y_group_offset + local_j;
    as_t[t_i * M + t_j] = tile[local_j][local_i];
}

