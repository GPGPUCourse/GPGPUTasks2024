#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
        __global float *a,
        __global float *b,
        __global float *c,
        unsigned int M,
        unsigned int K,
        unsigned int N
        )
{
    int i = get_global_id(0);   // до N
    int j = get_global_id(1);   // до M

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += a[j * K + k] * b[k * N + i];
    }

    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
        __global float *a,
        __global float *b,
        __global float *c,
        unsigned int M,
        unsigned int K,
        unsigned int N
        )
{
    int i = get_global_id(0);   // до N
    int j = get_global_id(1);   // до M

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (unsigned int tile_start_i = 0; tile_start_i < K; tile_start_i += TILE_SIZE) {
        if (tile_start_i + local_i < K && tile_start_i + local_j < K) {
            tileA[local_j][local_i] = a[j * K + tile_start_i + local_i];
            tileB[local_j][local_i] = b[(tile_start_i + local_j) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            if (tile_start_i + k >= K) {
                continue;
            }
            sum += tileA[local_j][k] * tileB[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
        __global float *a,
        __global float *b,
        __global float *c,
        unsigned int M,
        unsigned int K,
        unsigned int N
        )
{
    int i = get_global_id(0) * WORK_PER_THREAD;   // до N / wpt
    int j = get_global_id(1);   // до M

    int local_i = get_local_id(0) * WORK_PER_THREAD; // до N / wpt
    int local_j = get_local_id(1); // до M

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];

    for (unsigned int wpt_i = 0; wpt_i < WORK_PER_THREAD; wpt_i++) {
        sum[wpt_i] = 0.0f;
    }

    for (unsigned int tile_start_i = 0; tile_start_i < K; tile_start_i += TILE_SIZE) {
        if (tile_start_i + local_i < K && tile_start_i + local_j < K) {
            for (unsigned int i_wpt = 0; i_wpt < WORK_PER_THREAD; i_wpt++) {
                tileA[local_j][local_i + i_wpt] = a[j * K + tile_start_i + (local_i + i_wpt)];
                tileB[local_j][local_i + i_wpt] = b[(tile_start_i + local_j) * N + i + i_wpt];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            for (unsigned int wpt_i = 0; wpt_i < WORK_PER_THREAD; wpt_i++) {
                sum[wpt_i] += tileA[local_j][k] * tileB[k][local_i + wpt_i];;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int wpt_i = 0; wpt_i < WORK_PER_THREAD; wpt_i++) {
         c[j * N + i + wpt_i] = sum[wpt_i];
    }
}
#endif
