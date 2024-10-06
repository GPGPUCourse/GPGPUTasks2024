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
        sum += as[i * K + k] * bs[k * N + j];
    }
    cs[i * N + j] = sum;
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
        tile_as[li][lj] = as[gi * K + lj + (tile_k * TILE_SIZE)];
        tile_bs[li][lj] = bs[(li + (tile_k * TILE_SIZE)) * N + gj];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < TILE_SIZE; t++) {
            sum += tile_as[li][t] * tile_bs[t][lj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[gi * N + gj] = sum;
}

#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void
matrix_multiplication_local_wpt(__global float *as, __global float *bs, __global float *cs, unsigned int M,
                                unsigned int K,
                                unsigned int N) {
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);

    __local float tile_as[TILE_SIZE][TILE_SIZE];
    __local float tile_bs[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD] = {0.f};
    for (int tile_k = 0; tile_k * TILE_SIZE < K; tile_k++) {
        for (int wi = 0; wi < WORK_PER_THREAD; wi++) {

            tile_as[li * WORK_PER_THREAD + wi][lj] = as[(gi * WORK_PER_THREAD + wi) * K + lj +
                                                        (tile_k * TILE_SIZE)];
            tile_bs[li * WORK_PER_THREAD + wi][lj] = bs[((li * WORK_PER_THREAD + wi) + (tile_k * TILE_SIZE)) * N +
                                                        gj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
            for (int t = 0; t < TILE_SIZE; t++) {
                sum[wi] += tile_as[li * WORK_PER_THREAD + wi][t] * tile_bs[t][lj];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    for (int wi = 0; wi < WORK_PER_THREAD; wi++) {
        cs[(gi * WORK_PER_THREAD + wi) * N + gj] = sum[wi];
    }
}

#endif
