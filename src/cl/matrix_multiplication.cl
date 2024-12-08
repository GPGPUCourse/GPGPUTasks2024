#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float const * const a,
                                          __global float const * const b,
                                          __global float * const c,
                                          unsigned int const M,
                                          unsigned int const K,
                                          unsigned int const N)
{
    unsigned int const i = get_global_id(0);
    unsigned int const j = get_global_id(1);

    if (i >= N || j >= M) {
        return;
    }

    float sum = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        sum += a[j * K + k] * b[k * N + i];
    }

    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float const * const a,
                                          __global float const * const b,
                                          __global float * const c,
                                          unsigned int const M,
                                          unsigned int const K,
                                          unsigned int const N)
{
    unsigned int const i = get_global_id(0);
    unsigned int const j = get_global_id(1);

    unsigned int const local_i = get_local_id(0);
    unsigned int const local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    float sum = 0.f;
    for (unsigned int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = a[tileK * TILE_SIZE + j * K + local_i];
        tileB[local_j][local_i] = b[tileK * TILE_SIZE * N + local_j * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int t = 0; t < TILE_SIZE; ++t) {
            sum += tileA[local_j][t] * tileB[t][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < N && j < M) {
        c[N * j + i] = sum;
    }
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void
matrix_multiplication_local_wpt(__global float const * const a,
                                __global float const * const b,
                                __global float * const c,
                                unsigned int const M,
                                unsigned int const K,
                                unsigned int const N) {
    unsigned int const rts = TILE_SIZE / WORK_PER_THREAD;                       
    unsigned int const local_i = get_local_id(0);
    unsigned int const local_j = get_local_id(1);
    unsigned int const i = TILE_SIZE * get_group_id(0) + local_i;
    unsigned int const j = TILE_SIZE * get_group_id(1) + local_j;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD] = {0.f};
        for (unsigned int k = 0; k < WORK_PER_THREAD; ++k) {
        sum[k] = 0.0f;
    }
    
    for (unsigned int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        for (unsigned int wi = 0; wi < WORK_PER_THREAD; ++wi) {
            unsigned int const i_a = tileK * TILE_SIZE + local_i;
            unsigned int const j_a = j + wi * rts;
            tileA[local_j + wi * rts][local_i] = a[j_a * K + i_a];
            	
            unsigned int const i_b = i;
            unsigned int const j_b = tileK * TILE_SIZE + local_j;
            tileB[local_j + wi * rts][local_i] = b[(j_b + wi * rts) * N + i_b];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int wi = 0; wi < WORK_PER_THREAD; ++wi) {
            for (unsigned int t = 0; t < TILE_SIZE; t++) {
                sum[wi] += tileA[local_j + rts * wi][t] * tileB[t][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    for (unsigned int wi = 0; wi < WORK_PER_THREAD; ++wi) {
        c[(j + wi * rts) * N + i] = sum[wi];
    }
}

#endif
