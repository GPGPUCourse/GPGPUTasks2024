#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
        __global float *as,
        __global float *bs,
        __global float *cs,
        const unsigned int M,
        const unsigned int K,
        const unsigned int N
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= N || j >= M)
        return;

    float sum = 0.0f;
    for (unsigned int k = 0; k < K; k++)
        sum += as[j * K + k] * bs[k * N + i];
    cs[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
        __global float *as,
        __global float *bs,
        __global float *cs,
        const unsigned int M,
        const unsigned int K,
        const unsigned int N
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
        if (i < N && j < M && tile_start + local_i < K)
            tile_a[local_j][local_i] = as[(tile_start + local_i) + j * K];
        else
            tile_a[local_j][local_i] = 0.0f;

        if (i < N && j < M && tile_start + local_j < K)
            tile_b[local_j][local_i] = bs[i + (tile_start + local_j) * K];
        else
            tile_b[local_j][local_i] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; ++l)
            sum += tile_a[local_j][l] * tile_b[l][local_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < N && j < M)
        cs[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
        __global float *as,
        __global float *bs,
        __global float *cs,
        const unsigned int M,
        const unsigned int K,
        const unsigned int N
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
        if (i < N && j < M && tile_start + local_i < K)
            tile_a[local_j][local_i] = as[(tile_start + local_i) + j * K];
        else
            tile_a[local_j][local_i] = 0.0f;

        if (i < N && j < M && tile_start + local_j < K)
            tile_b[local_j][local_i] = bs[i + (tile_start + local_j) * N];
        else
            tile_b[local_j][local_i] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; ++l)
            sum += tile_a[local_j][l] * tile_b[l][local_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < N && j < M)
        cs[j * N + i] = sum;
}
#endif
