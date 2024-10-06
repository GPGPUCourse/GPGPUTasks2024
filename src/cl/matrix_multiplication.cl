#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global float *a,
    __global float *b,
    __global float *c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= N || j >= M) {
        return;
    }
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global float *a,
    __global float *b,
    __global float *c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= N || j >= M) {
        return;
    }
    
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * K + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global float *a,
    __global float *b,
    __global float *c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int i = get_global_id(0);
    int j = get_global_id(1) * WORK_PER_THREAD;

    if (i >= N || j >= M) {
        return;
    }

    int local_i = get_local_id(0);
    int local_j = get_local_id(1) * WORK_PER_THREAD;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        sum[w] = 0.0f;
    }

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            tileA[local_j + w][local_i] = a[(j + w) * K + (tileK * TILE_SIZE + local_i)];
            tileB[local_j + w][local_i] = b[(tileK * TILE_SIZE + local_j + w) * K + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                sum[w] += tileA[local_j + w][k] * tileB[k][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        c[(j + w) * N + i] = sum[w];
    }
}
#endif
