#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= k || j >= m)
        return;
    float x = a[j * k + i];
    at[i * m + j] = x;
}

#define ONE_DIMENSION_SIZE 16
__kernel void matrix_transpose_local_bad_banks(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);
    __local float buffer[ONE_DIMENSION_SIZE][ONE_DIMENSION_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    if (global_i < k && global_j < m)
        buffer[local_j][local_i] = a[global_j * k + global_i];
    else
        buffer[local_j][local_i] = 0;
    float tmp = buffer[local_j][local_i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_i <= local_j) {
        buffer[local_j][local_i] = buffer[local_i][local_j];
        buffer[local_i][local_j] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int new_i = global_j / ONE_DIMENSION_SIZE * ONE_DIMENSION_SIZE;
    int new_j = global_i / ONE_DIMENSION_SIZE * ONE_DIMENSION_SIZE;
    if (new_i + local_i < m && new_j + local_j < k)
        at[(new_j + local_j) * m + new_i + local_i] = buffer[local_j][local_i];
}

__kernel void matrix_transpose_local_good_banks(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);
    __local float buffer[(ONE_DIMENSION_SIZE + 1) * ONE_DIMENSION_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (global_i < k && global_j < m)
        buffer[local_j * (ONE_DIMENSION_SIZE + 1) + local_i] = a[global_j * k + global_i];
    else
        buffer[local_j * (ONE_DIMENSION_SIZE + 1) + local_i] = 0;
    float tmp = buffer[local_j * (ONE_DIMENSION_SIZE + 1) + local_i];
    barrier(CLK_LOCAL_MEM_FENCE);
    buffer[local_i * (ONE_DIMENSION_SIZE + 1) + local_j] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
    int new_i = global_j / ONE_DIMENSION_SIZE * ONE_DIMENSION_SIZE;
    int new_j = global_i / ONE_DIMENSION_SIZE * ONE_DIMENSION_SIZE;
    if (new_i + local_i < m && new_j + local_j < k)
        at[(new_j + local_j) * m + new_i + local_i] = buffer[local_j * (ONE_DIMENSION_SIZE + 1) + local_i];
}
