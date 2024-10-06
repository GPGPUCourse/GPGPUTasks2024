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
__kernel void matrix_multiplication_local_wpt(__global const float* A, __global const float* B, __global float* C, const unsigned int M, const unsigned int N, const unsigned int K) {
    __local float A_tile[TILE_SIZE][TILE_SIZE + 1];
    __local float B_tile[TILE_SIZE][TILE_SIZE + 1];

    int i = get_global_id(0) * WORK_PER_THREAD;
    int j = get_global_id(1) * WORK_PER_THREAD;
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    float sum[WORK_PER_THREAD][WORK_PER_THREAD] = {{0.0f}};

    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        for (int wi = 0; wi < WORK_PER_THREAD; ++wi) {
            for (int wj = 0; wj < WORK_PER_THREAD; ++wj) {
                if (i + wi < M && k * TILE_SIZE + local_j < K) {
                    A_tile[local_i + wi][local_j] = A[(i + wi) * K + k * TILE_SIZE + local_j];
                } else {
                    A_tile[local_i + wi][local_j] = 0.0f;
                }
                if (j + wj < N && k * TILE_SIZE + local_i < K) {
                    B_tile[local_i][local_j + wj] = B[(k * TILE_SIZE + local_i) * N + (j + wj)];
                } else {
                    B_tile[local_i][local_j + wj] = 0.0f;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int wi = 0; wi < WORK_PER_THREAD; ++wi) {
            for (int wj = 0; wj < WORK_PER_THREAD; ++wj) {
                for (int t = 0; t < TILE_SIZE; ++t) {
                    sum[wi][wj] += A_tile[local_i + wi][t] * B_tile[t][local_j + wj];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wi = 0; wi < WORK_PER_THREAD; ++wi) {
        for (int wj = 0; wj < WORK_PER_THREAD; ++wj) {
            if (i + wi < M && j + wj < N) {
                C[(i + wi) * N + (j + wj)] = sum[wi][wj];
            }
        }
    }
}
#endif
