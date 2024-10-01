#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a,
                                          __global float *b,
                                          __global float *c,
                                          unsigned int M,
                                          unsigned int K,
                                          unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= M || j >= N) {
        return;
    }

    float sum = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        sum += a[K * i + k] * b[N * k + j];
    }

    c[N * i + j] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a,
                                          __global float *b,
                                          __global float *c,
                                          unsigned int M,
                                          unsigned int K,
                                          unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0;

    for (unsigned int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; k_tile++) {
       unsigned i_a = i;
       unsigned j_a = k_tile * TILE_SIZE + local_j;
       tile_a[local_i][local_j] = i_a < M && j_a < K ? a[K * i_a + j_a] : 0.0f;

       unsigned i_b = k_tile * TILE_SIZE + local_i;
       unsigned j_b = j;
       tile_b[local_i][local_j] = i_b < K && j_b < N ? b[N * i_b + j_b] : 0.0f;

       barrier(CLK_LOCAL_MEM_FENCE);

       for (unsigned int k = 0; k < TILE_SIZE; k++) {
           sum += tile_a[local_i][k] * tile_b[k][local_j];
       }

       barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < M && j < N) {
        c[N * i + j] = sum;
    }
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a,
                                              __global float *b,
                                              __global float *c,
                                              unsigned int M,
                                              unsigned int K,
                                              unsigned int N)
{
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int i = TILE_SIZE * get_group_id(0) + local_i;
    unsigned int j = TILE_SIZE * get_group_id(1) + local_j;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        sum[w] = 0.0f;
    }

    const unsigned int WORK_GROUP_SIZE_X = TILE_SIZE / WORK_PER_THREAD;

    for (unsigned int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
            unsigned int i_a = i;
            unsigned int j_a = (TILE_SIZE * t + local_j) + WORK_GROUP_SIZE_X * w;
            unsigned int i_b = TILE_SIZE * t + local_i;
            unsigned int j_b = j + WORK_GROUP_SIZE_X * w;
            tile_a[local_i][WORK_GROUP_SIZE_X * w + local_j] = i_a < M && j_a < K ? a[K * i_a + j_a] : 0.0f;
            tile_b[local_i][WORK_GROUP_SIZE_X * w + local_j] = i_b < K && j_b < N ? b[N * i_b + j_b] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
                float tmp = tile_a[local_i][k];
                sum[w] += tmp * tile_b[k][WORK_GROUP_SIZE_X * w + local_j];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        if (i < M && (WORK_GROUP_SIZE_X * w + j) < N) {
            c[N * i + (WORK_GROUP_SIZE_X * w + j)] = sum[w];
        }
    }
}
#endif
