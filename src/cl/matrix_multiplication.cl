#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
        __global const float *as,
        __global const float *bs,
        __global float *cs,
        unsigned int m,
        unsigned int k,
        unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= m || j >= n) return;

    float res = 0.0f;
    for (int l = 0; l < k; ++l) {
        res += as[j * k + l] * bs[l * n + i];
    }
    cs[j * n + i] = res;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
        __global const float *as,
        __global const float *bs,
        __global float *cs,
        unsigned int m,
        unsigned int k,
        unsigned int n
) {
    unsigned int wi = get_global_id(0);
    unsigned int wj = get_global_id(1);
    unsigned int i = get_local_id(0);
    unsigned int j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float res = 0.0f;
    for (int tileOff = 0; tileOff < k; tileOff += TILE_SIZE) {

        barrier(CLK_LOCAL_MEM_FENCE);
        tileA[j][i] = wi >= n || wj >= m || tileOff + i >= k ? 0 : as[(tileOff + i) + wj * k];
        tileB[j][i] = wi >= n || wj >= m || tileOff + j >= k ? 0 : bs[wi + (tileOff + j) * k];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; ++l) {
            res += tileA[j][l] * tileB[l][i];
        }
    }
    if (wi < n && wj < m) {
        cs[wj * n + wi] = res;
    }
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
        __global const float *as,
        __global const float *bs,
        __global float *cs,
        unsigned int m,
        unsigned int k,
        unsigned int n
) {
    unsigned int i = get_local_id(0);
    unsigned int j = get_local_id(1) * WORK_PER_THREAD;
    unsigned int wi = get_group_id(0) * TILE_SIZE + i;
    unsigned int wj = get_group_id(1) * TILE_SIZE + j;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float res[WORK_PER_THREAD] = {};
    for (int dj = 0; dj < WORK_PER_THREAD; ++dj) res[dj] = 0;

    for (int tileOff = 0; tileOff < k; tileOff += TILE_SIZE) {

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int dj = 0; dj < WORK_PER_THREAD; ++dj) {
            tileA[j + dj][i] = wi >= n || wj >= m || tileOff + i >= k ? 0 : as[(tileOff + i) + (wj + dj) * k];
            tileB[j + dj][i] = wi >= n || wj >= m || tileOff + j >= k ? 0 : bs[wi + (tileOff + j + dj) * n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; ++l) {
            for (int dj = 0; dj < WORK_PER_THREAD; ++dj) {
                res[dj] += tileA[j + dj][l] * tileB[l][i];
            }
        }
    }
    for (int dj = 0; dj < WORK_PER_THREAD; ++dj) {
        if (wi < n && wj + dj < m) {
            cs[(wj + dj) * n + wi] = res[dj];
        }
    }
}
#endif
