#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global const float* as,
    __global const float* bs,
    __global float* cs,
    int M,
    int K,
    int N)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);

    float res = 0;
    for (int k = 0; k < K; ++k) {
        res += as[gid0 * K + k] * bs[k * N + gid1];
    }

    cs[gid0 * N + gid1] = res;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global const float* as,
    __global const float* bs,
    __global float* cs,
    int M,
    int K,
    int N)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);

    __local float abuf[TILE_SIZE][TILE_SIZE];
    __local float bbuf[TILE_SIZE][TILE_SIZE];
    
    float res = 0;
    for (int i = 0; i < K; i += TILE_SIZE) {
        // load tiles to buffers
        abuf[lid1][lid0] = as[gid1 * K + (i + lid0)];
        bbuf[lid1][lid0] = bs[(i + lid1) * N + gid0];

        barrier(CLK_LOCAL_MEM_FENCE);

        // accumulate results into res
        for (int k = 0; k < TILE_SIZE; ++k) {
            res += abuf[lid1][k] * bbuf[k][lid0];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[gid0 + gid1 * N] = res;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global const float* as,
    __global const float* bs,
    __global float* cs,
    int M,
    int K,
    int N)
{
    int lid0 = get_local_id(0) * WORK_PER_THREAD;
    int lid1 = get_local_id(1);
    int gid0 = get_global_id(0) * WORK_PER_THREAD;
    int gid1 = get_global_id(1);

    __local float abuf[TILE_SIZE][TILE_SIZE];
    __local float bbuf[TILE_SIZE][TILE_SIZE];
    
    float res[WORK_PER_THREAD];
    for (int j = 0; j < WORK_PER_THREAD; ++j) {
        res[j] = 0;
    }

    for (int i = 0; i < K; i += TILE_SIZE) {
        // load tiles to buffers
        for (int j = 0; j < WORK_PER_THREAD; ++j) {
            abuf[lid0 + j][lid1] = as[(gid0 + j) * K + (i + lid1)];
            bbuf[lid0 + j][lid1] = bs[(i + lid0 + j) * N + gid1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // accumulate results into res
        for (int j = 0; j < WORK_PER_THREAD; ++j) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                res[j] += abuf[lid0 + j][k] * bbuf[k][lid1];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int j = 0; j < WORK_PER_THREAD; ++j) {
        cs[gid1 + (gid0 + j) * N] = res[j];
    }
}
#endif
