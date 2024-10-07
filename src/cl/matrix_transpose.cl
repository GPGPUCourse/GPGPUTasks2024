#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global   const float* a,
                                     __global   float* at,
                                     /* rows */ unsigned int m,
                                     /* cols */ unsigned int k)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= k || row >= m) {
        return;
    }

    float x = a[row * k + col];
    at[col * m + row] = x;
}

#define TILE_SIZE 16
__kernel void matrix_transpose_local_bad_banks(__global   const float* a,
                                               __global   float* at,
                                               /* rows */ unsigned int m,
                                               /* cols */ unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= k || j >= m) {
        return;
    }

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float buf[TILE_SIZE][TILE_SIZE];

    buf[local_i][local_j] = a[i * k + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    at[j * m + i] = buf[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(__global   const float* a,
                                                __global   float* at,
                                                /* rows */ unsigned int m,
                                                /* cols */ unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= k || j >= m) {
        return;
    }

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float buf[TILE_SIZE + 1][TILE_SIZE];

    buf[local_i][local_j] = a[i * k + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    at[j * m + i] = buf[local_i][local_j];
}
