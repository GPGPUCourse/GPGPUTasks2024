#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float *a, 
                                     __global float *a_t, 
                                     const int m, 
                                     const int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    a_t[i * m + j] = a[j * k + i];
}

#define SIZE 16

__kernel void matrix_transpose_local_bad_banks(__global float *a,
                                               __global float *a_t, unsigned int m, unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int local_diff = local_i - local_j;

    __local float buffer[SIZE][SIZE];
    buffer[local_i][local_j] = a[j * m + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    a_t[(i - local_diff) * m + j + (local_diff)] = buffer[local_j][local_i];
}


__kernel void matrix_transpose_local_good_banks(__global float *a, 
                                                __global float *a_t, 
                                                const int m, 
                                                const int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int local_diff = local_i - local_j;

    __local float buffer[SIZE][SIZE + 1];
    buffer[local_i][local_j] = a[j * m + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    a_t[(i - local_diff) * m + j + (local_diff)] = buffer[local_j][local_i];
}

