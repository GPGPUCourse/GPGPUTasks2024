#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= N || j >= M)
        return;
    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += a[j * K + k] * b[k * N + i];
    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE]; //// SHIFT
    float sum = 0.f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        if (global_j < M && (tileK * TILE_SIZE + local_i) < K)
            tileA[local_j][local_i] = a[global_j * K + tileK * TILE_SIZE + local_i];
        else
            tileA[local_j][local_i] = 0.;
        if (global_i < N && (tileK * TILE_SIZE + local_j) < K)
            tileB[local_j][local_i] = b[global_i + (tileK * TILE_SIZE + local_j) * N];
        else
            tileB[local_j][local_i] = 0.;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_j < M && global_i < N)
        c[global_j * N + global_i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    
    int idX = get_group_id(0);
    int idY = get_group_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE]; //// SHIFT
    float sum[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; ++w)
        sum[w] = 0.;
    
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            if ((idY * TILE_SIZE + local_j * WORK_PER_THREAD + w) < M && tileK * TILE_SIZE + local_i < K)
                tileA[local_j * WORK_PER_THREAD + w][local_i] = a[(idY * TILE_SIZE + local_j * WORK_PER_THREAD + w) * K + tileK * TILE_SIZE + local_i];
            else
                tileA[local_j * WORK_PER_THREAD + w][local_i] = 0.;
            if (global_i < N && (tileK * TILE_SIZE + local_j * WORK_PER_THREAD + w) < K)
                tileB[local_j * WORK_PER_THREAD + w][local_i] = b[global_i + (tileK * TILE_SIZE + local_j * WORK_PER_THREAD + w) * N];
            else
                tileB[local_j * WORK_PER_THREAD + w][local_i] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // if (global_i == 0 && global_j == 0 && WORK_PER_THREAD > 2 && tileK == 4)
        //     printf("OK. tileK %d\n", tileK);////
        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                // if (global_i == 0 && global_j == 0 && WORK_PER_THREAD > 2 && tileK == 4)
                //     printf("OK. tileK %d k: %d w: %d current_i: %d current_j: %d\n", tileK, k, w, local_i, local_j + w * WORK_PER_THREAD);////
                sum[w] += tileA[local_j * WORK_PER_THREAD + w][k] * tileB[k][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WORK_PER_THREAD; ++w)
        if (idY * TILE_SIZE + local_j * WORK_PER_THREAD + w < M && global_i < N)
        c[(idY * TILE_SIZE + local_j * WORK_PER_THREAD + w) * N + global_i] = sum[w];
}
#endif
