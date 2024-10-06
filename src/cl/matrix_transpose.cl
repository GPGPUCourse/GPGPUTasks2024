#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(
    __global const float* A,
    __global float* A_T,
    const unsigned int M,
    const unsigned int K)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < M && j < K) {
        A_T[j * M + i] = A[i * K + j];
    }
}

__kernel void matrix_transpose_local_bad_banks(
    __global const float* A,
    __global float* A_T,
    const unsigned int M,
    const unsigned int K)
{
    __local float tile[32][32];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < M && j < K) {
        tile[local_j][local_i] = A[i * K + j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < M && j < K) {
        A_T[j * M + i] = tile[local_i][local_j];
    }
}


__kernel void matrix_transpose_local_good_banks(
    __global const float* A,
    __global float* A_T,
    const unsigned int M,
    const unsigned int K)
{
    __local float tile[32][32 + 1];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < M && j < K) {
        tile[local_j][local_i] = A[i * K + j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < M && j < K) {
        A_T[j * M + i] = tile[local_i][local_j];
    }
}
