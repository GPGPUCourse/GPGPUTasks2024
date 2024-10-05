#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = a[j * k + i];
    at[i * m + j] = x;
}

#define SIZE 16
__kernel void matrix_transpose_local_bad_banks(__global float* a, __global float* at, unsigned int m, unsigned int k)
{
    __local float buffer[SIZE * SIZE];
    int i = get_global_id(0);
    int j = get_global_id(1);
    int i_local = get_local_id(0);
    int j_local = get_local_id(1);

    buffer[i_local * SIZE + j_local] = a[i * k + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    at[j * m + i] = buffer[i_local * SIZE + j_local];
}

__kernel void matrix_transpose_local_good_banks(__global float* a, __global float* at, unsigned int m, unsigned int k)
{
    __local float buffer[SIZE * (SIZE + 1)];
    int i = get_global_id(0);
    int j = get_global_id(1);
    int i_local = get_local_id(0);
    int j_local = get_local_id(1);

    buffer[j_local * (SIZE + 1) + i_local] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    at[j * m + i] = buffer[j_local * (SIZE + 1) + i_local];
}