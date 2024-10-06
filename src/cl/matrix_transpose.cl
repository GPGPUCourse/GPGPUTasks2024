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
                                               __global float *a_t, 
                                               const int m, 
                                               const int k)
{
    __local float buffer[SIZE * SIZE];
    int i = get_global_id(0);
    int j = get_global_id(1);
    int l_i = get_local_id(0);
    int l_j = get_local_id(1);
    buffer[l_i * SIZE + l_j] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    a_t[m * j + i] = buffer[l_i * SIZE + l_j];
}

__kernel void matrix_transpose_local_good_banks(__global float *a, 
                                               __global float *a_t, 
                                               const int m, 
                                               const int k)
{
    __local float buffer[SIZE * (SIZE + 1)];
    int i = get_global_id(0);
    int j = get_global_id(1);
    int l_i = get_local_id(0);
    int l_j = get_local_id(1);
    buffer[l_j * (SIZE + 1) + l_i] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    a_t[m * j + i] = buffer[l_j * (SIZE + 1) + l_i];
}
