#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global const float* A, __global const float* B, __global float* C, const unsigned int M, const unsigned int N, const unsigned int K) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global const float* A, __global const float* B, __global float* C, const unsigned int M, const unsigned int N, const unsigned int K) {
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    float sum = 0.0f;

    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (i < M && k * TILE_SIZE + local_j < K) {
            A_tile[local_i][local_j] = (k * TILE_SIZE + local_j < K) ? A[i * K + k * TILE_SIZE + local_j] : 0.0f;
        }
        if (j < N && k * TILE_SIZE + local_i < K) {
            B_tile[local_i][local_j] = (k * TILE_SIZE + local_i < K) ? B[(k * TILE_SIZE + local_i) * N + j] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < TILE_SIZE; ++t) {
            sum += A_tile[local_i][t] * B_tile[t][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < M && j < N) {
        C[i * N + j] = sum;
    }
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *A, 
                                              __global float *B, 
                                              __global float *C, 
                                              const int M, 
                                              const int K, 
                                              const int N) 
{
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1) * WORK_PER_THREAD;
    const unsigned int local_j = get_local_id(0);
    const unsigned int local_i = get_local_id(1) * WORK_PER_THREAD;

    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (int p = 0; p < WORK_PER_THREAD; p++) {
        sum[p] = 0;
    }
    for (int t = 0; t * TILE_SIZE < K; t++) {
        for (int p = 0; p < WORK_PER_THREAD; p++) {
            A_tile[local_i + p][local_j] = A[(i + p) * K + (t * TILE_SIZE + local_j)];
            B_tile[local_i + p][local_j] = B[(t * TILE_SIZE + local_i + p) * N + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            const float tmp = B_tile[k][local_j];
            for (int r = 0; r < WORK_PER_THREAD; r++) {
                sum[r] += A_tile[local_i + r][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int p = 0; p < WORK_PER_THREAD; p++) {
        C[(i + p) * N + j] = sum[p];
    }
}
#endif
