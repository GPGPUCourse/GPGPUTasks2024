#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6
// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global const float* a,
    __global const float* b,
    __global       float* c,
    unsigned int M,
    unsigned int K,
    unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    float res = 0.0f;
    for (unsigned int l = 0; l < K; ++l) {
        res += a[j * K + l] * b[l * N + i];
    }
    c[j * N + i] = res;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global const float* a,
    __global const float* b,
    __global       float* c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int i = get_global_id(0);       // номер столбца результирующей C
    int j = get_global_id(1);       // номер строки результирующей C
    int i_local = get_local_id(0);  // номер столбца в tile
    int j_local = get_local_id(1);  // номер строки в tile

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    if (i >= M || j >= N) return;

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        int global_tile_i = tileK * TILE_SIZE + i_local;
        int global_tile_j = tileK * TILE_SIZE + j_local;

        if (global_tile_i < M && global_tile_j < K) {
            tileA[i_local][j_local] = a[i * K + global_tile_j];
        }

        if (global_tile_j < K && global_tile_i < N) {
            tileB[i_local][j_local] = b[global_tile_i * N + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TILE_SIZE; ++p) {
            sum += tileA[i_local][p] * tileB[p][j_local];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[i * N + j] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global const float* a,
    __global const float* b,
    __global       float* c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float acc[WORK_PER_THREAD] = { 0 };

    for (int t = 0; t < numTiles; ++t)
    {
        for (int w = 0; w < WORK_PER_THREAD; ++w)
        {
            tileA[local_j * WORK_PER_THREAD + w][local_i] = a[(j * WORK_PER_THREAD + w) * K + t * TILE_SIZE + local_i];
            tileB[local_j * WORK_PER_THREAD + w][local_i] = b[(t * TILE_SIZE + local_j * WORK_PER_THREAD + w) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i)
        {
            for (int w = 0; w < WORK_PER_THREAD; ++w)
            {
                acc[w] += tileA[local_j * WORK_PER_THREAD + w][i] * tileB[i][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WORK_PER_THREAD; ++w)
        c[(j * WORK_PER_THREAD + w) * N + i] = acc[w];
}
#endif
