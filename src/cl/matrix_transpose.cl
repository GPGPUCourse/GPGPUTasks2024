#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(
    __global const float *a,
    __global float *a_t,
    unsigned int M, 
    unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= M || j >= K) {
        return;
    }

    a_t[j * K + i] = a[i * M + j];
}

__kernel void matrix_transpose_local_bad_banks(
    __global const float *a,
    __global float *a_t,
    unsigned int M, 
    unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    tile[local_j][local_i] = a[j * M + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int t_i = get_group_id(0) * TILE_SIZE + local_j;
    unsigned int t_j = get_group_id(1) * TILE_SIZE + local_i;
    
    a_t[t_i * K + t_j] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(
    __global const float *a,
    __global float *a_t,
    unsigned int M, 
    unsigned int K
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    tile[local_j][local_i] = a[j * M + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int t_i = get_group_id(0) * TILE_SIZE + local_j;
    unsigned int t_j = get_group_id(1) * TILE_SIZE + local_i;
    
    a_t[t_i * K + t_j] = tile[local_i][local_j];
}
