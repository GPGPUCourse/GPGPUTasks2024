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

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        A_tile[local_j][local_i] = A[j * K + local_i + k * TILE_SIZE];
        B_tile[local_j][local_i] = B[(k * TILE_SIZE + local_j) * N + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < TILE_SIZE; ++t) {
            sum += A_tile[local_j][t] * B_tile[t][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *A, __global float *B, __global float *C, unsigned int M,
                                              unsigned int K, unsigned int N) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD] = {0.f};
    for (int tile_k = 0; tile_k * TILE_SIZE < K; tile_k++) {
        for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
            A_tile[lj + wi * (TILE_SIZE / WORK_PER_THREAD)][li] = A[(gj + wi * (TILE_SIZE / WORK_PER_THREAD)) * K + li + (tile_k * TILE_SIZE)];
            B_tile[lj + wi * (TILE_SIZE / WORK_PER_THREAD)][li] = B[(lj + (tile_k * TILE_SIZE) + wi * (TILE_SIZE / WORK_PER_THREAD)) * N + gi];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
            for (int t = 0; t < TILE_SIZE; t++) {
                sum[wi] += A_tile[lj + wi * (TILE_SIZE / WORK_PER_THREAD)][t] * B_tile[t][li];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
        C[(gj + (TILE_SIZE / WORK_PER_THREAD) * wi) * N + gi] = sum[wi];
    }
}
#endif
