#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global const float* A,
    __global const float* B,
    __global float* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= N || j >= M) {
        return;
    }

    float sum = 0.0f;
    for (int t = 0; t < K; ++t) {
        sum += A[j * K + t] * B[t * N + i];
    }

    C[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global const float* A,
    __global const float* B,
    __global float* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        if (tile_k + local_i < K && j < M) {
            tileA[local_j][local_i] = A[j * K + (tile_k + local_i)];
        } else {
            tileA[local_j][local_i] = 0;
        }
        if (i < N && tile_k + local_j < K) {
            tileB[local_j][local_i] = B[(tile_k + local_j) * N + i];
        } else {
            tileB[local_j][local_i] = 0;
        }
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
__kernel void matrix_multiplication_local_wpt(
    __global const float* A,
    __global const float* B,
    __global float* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (int h = 0; h < WORK_PER_THREAD; ++h) {
        sum[h] = 0.0f;
    }

    const unsigned int local_j_base = local_j * WORK_PER_THREAD;
    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        for (int h = 0; h < WORK_PER_THREAD; ++h) {
            {
                const unsigned int ii = tile_k + local_i;
                const unsigned int jj = j * WORK_PER_THREAD + h;
                if (ii < K && jj < M) {
                    tileA[local_j_base + h][local_i] = A[jj * K + ii];
                } else {
                    tileA[local_j_base + h][local_i] = 0;
                }
            }

            {
                const unsigned int ii = i;
                const unsigned int jj = tile_k + local_j_base + h;
                if (ii < N && jj < K) {
                    tileB[local_j_base + h][local_i] = B[jj * K + ii];
                } else {
                    tileB[local_j_base + h][local_i] = 0;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            const float b = tileB[k][local_i];
            for (int h = 0; h < WORK_PER_THREAD; ++h) {
                sum[h] += b * tileA[local_j_base + h][k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int h = 0; h < WORK_PER_THREAD; ++h) {
        C[(j * WORK_PER_THREAD + h) * N + i] = sum[h];
    }
}
#endif
