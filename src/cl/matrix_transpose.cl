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

#define TILE_SIZE 32

__kernel void matrix_transpose_local_bad_banks(
    __global const float* A,
    __global float* A_T,
    const unsigned int M,
    const unsigned int K)
{
    __local float tile[TILE_SIZE][TILE_SIZE];
    
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[li][lj] = A[gj * M + gi];

    barrier(CLK_LOCAL_MEM_FENCE);

    A_T[(gi - li + lj) * M + (gj + li - lj)] = tile[lj][li];
}

__kernel void matrix_transpose_local_good_banks(
    __global const float* A,
    __global float* A_T,
    const unsigned int M,
    const unsigned int K)
{
    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    tile[li][lj] = A[gj * M + gi];

    barrier(CLK_LOCAL_MEM_FENCE);

    A_T[(gi - li + lj) * M + (gj + li - lj)] = tile[lj][li];
}
