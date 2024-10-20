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

        tile_a[local_j][local_i] = a[K * global_j + (t * TILE_SIZE + local_i)];
        tile_b[local_j][local_i] = b[N * (t * TILE_SIZE + local_j) + (global_i)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_j][k] * tile_b[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[N * global_j + global_i] = sum;
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
            tile_a[local_j + w * RTS][local_i] = a[(w * RTS + global_j) * K + (t * TILE_SIZE + local_i)];
            // for b
            tile_b[local_j + w * RTS][local_i] = b[(w * RTS + t * TILE_SIZE + local_j) * N + global_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
                float tmp = tile_a[w * RTS + local_j][k];
                acc[w] += tmp * tile_b[k][local_i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        c[global_i + N * (global_j + w * RTS)] = acc[w];
    }
};
#endif
