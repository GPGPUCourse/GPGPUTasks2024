#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define GROUP_SIZE (16)

__kernel void matrix_transpose_naive(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= M || j >= K) {
        return;
    }

    B[i * K + j] = A[j * M + i];
}

__kernel void matrix_transpose_local_bad_banks(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    __local float block[GROUP_SIZE][GROUP_SIZE];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);

    block[local_j][local_i] = A[j * M + i];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const unsigned int x = group_j * GROUP_SIZE + local_i;
    const unsigned int y = group_i * GROUP_SIZE + local_j;
    B[y * K + x] = block[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    __local float block[GROUP_SIZE][GROUP_SIZE + 1];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);

    block[local_j][local_i] = A[j * M + i];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const unsigned int x = group_j * GROUP_SIZE + local_i;
    const unsigned int y = group_i * GROUP_SIZE + local_j;
    B[y * K + x] = block[local_i][local_j];
}
