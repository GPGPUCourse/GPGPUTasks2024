#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global const float *A, __global const float *B, __global float *C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[j * K + k] * B[k * N + i];
    }

    C[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global const float *A, __global const float *B, __global float *C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i = get_group_id(0) * TILE_SIZE + local_i;
    int j = get_group_id(1) * TILE_SIZE + local_j;

    float sum = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = A[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = B[i + N * (tileK * TILE_SIZE + local_j)];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)

#define RTS (TILE_SIZE / WORK_PER_THREAD)

__kernel void matrix_multiplication_local_wpt(__global const float *A, __global const float *B, __global float *C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i = get_group_id(0) * TILE_SIZE + local_i;
    int j = get_group_id(1) * TILE_SIZE + local_j;

    float sum[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        sum[w] = 0.0f;
    }

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {

        for (int w = 0; w < WORK_PER_THREAD; w++) {
            tileA[local_j + w * RTS][local_i] = A[(j + w * RTS) * K + (tileK * TILE_SIZE + local_i)];
            tileB[local_j + w * RTS][local_i] = B[i + N * (tileK * TILE_SIZE + local_j + w * RTS)];        
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int w = 0; w < WORK_PER_THREAD; w++) {
                sum[w] += tileA[local_j + w * RTS][k] * tileB[k][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        C[(j + w * RTS) * N + i] = sum[w];
    }
}
#endif
