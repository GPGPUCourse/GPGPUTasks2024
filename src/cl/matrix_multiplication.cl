#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global float* A, // [M x K] - first is rows cnt, second columns
    __global float* B, // [K x N]
    __global float* C, // [M x N]
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    size_t gx = get_global_id(0); // in [0, N)
    size_t gy = get_global_id(1); // in [0, M)

    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += A[gy * K + k] * B[k * N + gx];
    }
    C[gy * N + gx] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global float* A, // [M x K] - first is rows cnt, second columns
    __global float* B, // [K x N]
    __global float* C, // [M x N]
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local float tile_A[TILE_SIZE][TILE_SIZE];
    __local float tile_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    for (int k = 0; k * TILE_SIZE < K; k++) {
        tile_A[lx][ly] = A[gx * K + ly + (k * TILE_SIZE)];
        tile_B[lx][ly] = B[(lx + k * TILE_SIZE) * N + gy];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_A[lx][j] * tile_B[j][ly];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[gx * N + gy] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global float* A, // [M x K] - first is rows cnt, second columns
    __global float* B, // [K x N]
    __global float* C, // [M x N]
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local float tile_A[TILE_SIZE][TILE_SIZE];
    __local float tile_B[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD] = { 0.0f };

    for (int k = 0; k * TILE_SIZE < K; k++) {
        for (int w = 0; w < WORK_PER_THREAD; w++) {

            tile_A[lx * WORK_PER_THREAD + w][ly] = A[(gx * WORK_PER_THREAD + w) * K + ly + (k * TILE_SIZE)];
            tile_B[lx * WORK_PER_THREAD + w][ly] = B[((lx * WORK_PER_THREAD + w) + (k * TILE_SIZE)) * N + gy];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int w = 0; w < WORK_PER_THREAD; w++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                sum[w] += tile_A[lx * WORK_PER_THREAD + w][j] * tile_B[j][ly];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        C[(gx * WORK_PER_THREAD + w) * N + gy] = sum[w];
    }
}
#endif
