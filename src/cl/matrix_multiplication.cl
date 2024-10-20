#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void
matrix_multiplication_naive(__global float *as, __global float *bs, __global float *cs, unsigned int M, unsigned int K,
                            unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.f;
    for (int k = 0; k < K; k++) {
        sum += as[j * K + k] * bs[k * N + i];
    }
    cs[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void
matrix_multiplication_local(__global float *as, __global float *bs, __global float *cs, unsigned int M, unsigned int K,
                            unsigned int N) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    __local float tile_as[TILE_SIZE][TILE_SIZE];
    __local float tile_bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    for (int tile_k = 0; tile_k * TILE_SIZE < K; tile_k++) {
        tile_as[lj][li] = as[gj * K + li + (tile_k * TILE_SIZE)];
        tile_bs[lj][li] = bs[(lj + (tile_k * TILE_SIZE)) * N + gi];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < TILE_SIZE; t++) {
            sum += tile_as[lj][t] * tile_bs[t][li];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[gj * N + gi] = sum;
}

#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)

#define RTS (TILE_SIZE / WORK_PER_THREAD)
__kernel void
matrix_multiplication_local_wpt(__global float *as, __global float *bs, __global float *cs, unsigned int M,
                                unsigned int K,
                                unsigned int N) {
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int gi = TILE_SIZE * get_group_id(0) + li;
    int gj = TILE_SIZE * get_group_id(1) + lj;

    __local float tile_as[TILE_SIZE][TILE_SIZE];
    __local float tile_bs[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD] = {0.f};
    for (int tile_k = 0; tile_k * TILE_SIZE < K; tile_k++) {
        for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
            int tile_i = TILE_SIZE * tile_k + li;
            int tile_j = TILE_SIZE * tile_k + lj;
            tile_as[lj + wi * RTS][li] = as[(gj + wi * RTS) * K + tile_i];
            tile_bs[lj + wi * RTS][li] = bs[(tile_j + wi * RTS) * N + gi];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
            for (int t = 0; t < TILE_SIZE; t++) {
                sum[wi] += tile_as[lj + RTS * wi][t] * tile_bs[t][li];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
        cs[(gj + wi * RTS) * N + gi] = sum[wi];
    }
}

#endif
