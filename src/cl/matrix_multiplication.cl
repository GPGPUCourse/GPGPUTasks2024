#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global const float* a,
                                          __global const float* b,
                                          __global float* c,
                                          unsigned int M,
                                          unsigned int K,
                                          unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= N || j >= M) {
        return;
    }

    int sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }

    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global const float* a,
                                          __global const float* b,
                                          __global float* c,
                                          unsigned int M,
                                          unsigned int K,
                                          unsigned int N)
{
    __local float local_a[TILE_SIZE][TILE_SIZE];
    __local float local_b[TILE_SIZE][TILE_SIZE];

    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    float sum = 0.0;
    for (int k = 0; k * TILE_SIZE < K; ++k) {
        local_a[local_j][local_i] = a[j * K + k * TILE_SIZE + local_i];
        local_b[local_j][local_i] = b[(k * TILE_SIZE + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k1 = 0; k1 < TILE_SIZE; ++k1) {
            sum += local_a[local_j][k1] * local_b[k1][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global const float* a,
                                              __global const float* b,
                                              __global float* c,
                                              unsigned int M,
                                              unsigned int K,
                                              unsigned int N)
{
    __local float local_a[TILE_SIZE][TILE_SIZE];
    __local float local_b[TILE_SIZE][TILE_SIZE];

    int i = get_global_id(0);
    int j = get_global_id(1) * WORK_PER_THREAD;

    int local_i = get_local_id(0);
    int local_j = get_local_id(1) * WORK_PER_THREAD ;

    float sum[WORK_PER_THREAD];
    for (int wpt = 0; wpt < WORK_PER_THREAD; ++wpt) {
        sum[wpt] = 0;
    }

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        for (int wpt = 0; wpt < WORK_PER_THREAD; ++wpt) {
            local_a[local_j + wpt][local_i] = a[(j + wpt) * K + k * TILE_SIZE + local_i];
            local_b[local_j + wpt][local_i] = b[(k * TILE_SIZE + local_j + wpt) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k1 = 0; k1 < TILE_SIZE; ++k1) {
            float loc_b = local_b[k1][local_i];
            for (int wpt = 0; wpt < WORK_PER_THREAD; ++wpt) {
                sum[wpt] += local_a[local_j + wpt][k1] * loc_b;
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wpt = 0; wpt < WORK_PER_THREAD; ++wpt) {
        c[(wpt + j) * N + i] = sum[wpt];
    }
}
#endif
