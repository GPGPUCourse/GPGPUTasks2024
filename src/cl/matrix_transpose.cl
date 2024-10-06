#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(
    __global float *a,
    __global float *at,
    unsigned int m,
    unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < k && j < m) {
        at[i * m + j] = a[j * k + i];
    }
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

    if (i < k && j < m) {
        tile[local_j][local_i] = a[j * k + i];
    } else {
        tile[local_j][local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int target_j = (i - local_i) + local_j;
    int target_i = (j - local_j) + local_i;
    if (target_i < k && target_j < m) {
        at[target_j * k + target_i] = tile[local_i][local_j];
    }
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

    if (i < k && j < m) {
        tile[local_j][local_i] = a[j * k + i];
    } else {
        tile[local_j][local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int target_j = (i - local_i) + local_j;
    int target_i = (j - local_j) + local_i;
    if (target_i < k && target_j < m) {
        at[target_j * k + target_i] = tile[local_i][local_j];
    }
}
