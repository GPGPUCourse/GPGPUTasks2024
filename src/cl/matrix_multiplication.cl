#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
        __global const float *a,
        __global const float *b,
        __global float *out,
        const unsigned int M,
        const unsigned int K,
        const unsigned int N
) {
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    if (i >= M || j >= N)
        return;

    float sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
    }

    out[i * N + j] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
        __global const float *a,
        __global const float *b,
        __global float *out,
        const unsigned int M,
        const unsigned int K,
        const unsigned int N
        )
{
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    const unsigned int jj = get_local_id(0);
    const unsigned int ii = get_local_id(1);

    __local float aa[TILE_SIZE][TILE_SIZE];
    __local float bb[TILE_SIZE][TILE_SIZE];
    float sum = 0;
    for (int tile = 0; tile * TILE_SIZE < K; ++tile) {

        if (i < M && (tile * TILE_SIZE + jj) < K)
            aa[ii][jj] = a[i * K + (tile * TILE_SIZE + jj)];
        else
            aa[ii][jj] = 0;

        if ((tile * TILE_SIZE + ii) < K && j < N)
            bb[ii][jj] = b[(tile * TILE_SIZE + ii) * N + j];
        else
            bb[ii][jj] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += aa[ii][k] * bb[k][jj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < M && j < N)
        out[i * N + j] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
        __global const float *a,
        __global const float *b,
        __global float *out,
        const unsigned int M,
        const unsigned int K,
        const unsigned int N
) {
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1) * WORK_PER_THREAD;

    const unsigned int jj = get_local_id(0);
    const unsigned int ii = get_local_id(1) * WORK_PER_THREAD;

    float sum[WORK_PER_THREAD];
    for (int q = 0; q < WORK_PER_THREAD; ++q) {
        sum[q] = 0;
    }

    __local float aa[TILE_SIZE][TILE_SIZE];
    __local float bb[TILE_SIZE][TILE_SIZE];

    for (int tile = 0; tile * TILE_SIZE < K; ++tile) {

        for (int q = 0; q < WORK_PER_THREAD; ++q) {
            if (i + q < M && (tile * TILE_SIZE + jj) < K)
                aa[ii + q][jj] = a[(i + q) * K + (tile * TILE_SIZE + jj)];
            else
                aa[ii + q][jj] = 0;

            if ((tile * TILE_SIZE + ii + q) < K && j < N)
                bb[ii + q][jj] = b[(tile * TILE_SIZE + ii + q) * N + j];
            else
                bb[ii + q][jj] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            const float b_cache = bb[k][jj];
            for (int q = 0; q < WORK_PER_THREAD; ++q) {
                sum[q] += aa[ii + q][k] * b_cache;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int q = 0; q < WORK_PER_THREAD; ++q) {
        if (i + q < M && j < N)
            out[(i + q) * N + j] = sum[q];
    }
}
#endif
