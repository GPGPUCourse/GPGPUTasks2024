#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global const float *as, __global float *as_t, unsigned int m, unsigned int k) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= k || j >= m)
        return;
    as_t[i * m + j] = as[j * k + i];
}

__kernel void matrix_transpose_local_bad_banks(__global const float *as, __global float *as_t, unsigned int m,
                                               unsigned int k) {
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    tile[local_j][local_i] = as[global_j * m + global_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[(get_group_id(0) * TILE_SIZE + local_j) * k + ( get_group_id(1) * TILE_SIZE + local_i)] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(__global float *as, __global float *as_t, unsigned int m,
                                                unsigned int k) {
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE + 1][TILE_SIZE];

    tile[local_j][local_i] = as[global_j * m + global_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[(get_group_id(0) * TILE_SIZE + local_j) * k + ( get_group_id(1) * TILE_SIZE + local_i)] = tile[local_i][local_j];
}
