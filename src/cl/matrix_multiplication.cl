#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге
#define MKN "MKN top"
__kernel void matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M,
                                          unsigned int K, unsigned N) {
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    if (global_i >= N || global_j >= M)
        return;

    c[N * global_j + global_i] = 0;
    for (int x = 0; x < K; ++x) {
        c[N * global_j + global_i] += a[global_j * K + x] * b[N * x + global_i];
    }
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M,
                                          unsigned int K, unsigned N) {
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (unsigned int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

        if (global_i < M && (t * TILE_SIZE + local_j) < K) {
            tile_a[local_i][local_j] = a[K * global_i + (t * TILE_SIZE + local_j)];
        } else {
            tile_a[local_i][local_j] = 0.f;
        }

        if ((t * TILE_SIZE + local_i) < K && global_j < N) {
            tile_b[local_i][local_j] = b[N * (t * TILE_SIZE + local_i) + global_j];
        } else {
            tile_b[local_i][local_j] = 0.f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_i][k] * tile_b[k][local_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_i < M && global_j < N) {
        c[N * global_i + global_j] = sum;
    }
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
    #define RTS TILE_SIZE / WORK_PER_THREAD
    #define NUM_TILES (K + TILE_SIZE - 1) / TILE_SIZE
__kernel void matrix_multiplication_local_wpt(__global float *a, __global float *b, __global float *c, unsigned int M,
                                              unsigned int K, unsigned int N) {
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int global_i = TILE_SIZE * get_group_id(0) + local_i;
    unsigned int global_j = TILE_SIZE * get_group_id(1) + local_j;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float acc[WORK_PER_THREAD];
    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        acc[w] = 0.0f;
    }

    for (unsigned int t = 0; t < NUM_TILES; t++) {
        for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
            // for a
            if (global_i >= M || (TILE_SIZE * t + local_j + RTS * w) >= K) {
                tile_a[local_i][RTS * w + local_j] = 0.f;
            } else {
                tile_a[local_i][RTS * w + local_j] = a[global_i * K + TILE_SIZE * t + local_j + w * RTS];
            }
            // for b
            if ((TILE_SIZE * t + local_i) >= K || (global_j + RTS * w) >= N ) {
                tile_b[local_i][local_j] = 0.f;
            } else {
                tile_b[local_i][RTS * w + local_j] = b[N * (t * TILE_SIZE+ local_i) + global_j + w * RTS];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
                float tmp = tile_a[local_i][k];
                acc[w] += tmp * tile_b[k][RTS * w + local_j];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        c[N * global_i + RTS * w + global_j] = acc[w];
    }
};
#endif
