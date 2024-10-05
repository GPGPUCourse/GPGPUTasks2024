#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    const unsigned int threadIdX = get_global_id(0);
    const unsigned int threadIdY = get_global_id(1);

    if (threadIdX >= M || threadIdY >= K) {
        return;
    }

    B[threadIdX * K + threadIdY] = A[threadIdY * M + threadIdX];
}

#define GROUP_SIZE (32)
__kernel void matrix_transpose_local_bad_banks(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    __local float block[GROUP_SIZE][GROUP_SIZE];

    const unsigned int threadIdX = get_global_id(0);
    const unsigned int threadIdY = get_global_id(1);
    const unsigned int localThreadIdX = get_local_id(0);
    const unsigned int localThreadIdY = get_local_id(1);

    const unsigned int groupIdX = get_group_id(0);
    const unsigned int groupIdY = get_group_id(1);

    block[localThreadIdY][localThreadIdX] = A[threadIdY * M + threadIdX];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const unsigned int x = groupIdY * GROUP_SIZE + localThreadIdX;
    const unsigned int y = groupIdX * GROUP_SIZE + localThreadIdY;
    B[y * K + x] = block[localThreadIdX][localThreadIdY];
}

__kernel void matrix_transpose_local_good_banks(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
        __local float block[GROUP_SIZE][GROUP_SIZE + 1];

    const unsigned int threadIdX = get_global_id(0);
    const unsigned int threadIdY = get_global_id(1);
    const unsigned int localThreadIdX = get_local_id(0);
    const unsigned int localThreadIdY = get_local_id(1);

    const unsigned int groupIdX = get_group_id(0);
    const unsigned int groupIdY = get_group_id(1);

    block[localThreadIdY][localThreadIdX] = A[threadIdY * M + threadIdX];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const unsigned int x = groupIdY * GROUP_SIZE + localThreadIdX;
    const unsigned int y = groupIdX * GROUP_SIZE + localThreadIdY;
    B[y * K + x] = block[localThreadIdX][localThreadIdY];
}
