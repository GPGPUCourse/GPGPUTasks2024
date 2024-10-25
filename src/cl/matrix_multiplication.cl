#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a,
                                          __global float *b, 
                                          __global float *c, 
                                          const int m, 
                                          const int k, 
                                          const int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < m && j < n) {
        float sum = 0;
        for (int p = 0; p < k; p++) {
            sum += a[i * k + p] * b[p * n + j];
        }
        c[i * n + j] = sum;
    }
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a,
                                          __global float *b, 
                                          __global float *c, 
                                          const int m, 
                                          const int k, 
                                          const int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    float sum = 0.0f;

    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t * TILE_SIZE < k; t++) {
        int t_tile_size = t * TILE_SIZE;
        a_tile[local_j][local_i] = a[j * k + local_i + t_tile_size];
        b_tile[local_j][local_i] = b[(local_j + t_tile_size) * n + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TILE_SIZE; p++) {
            sum += a_tile[local_j][p] * b_tile[p][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * n + i] = sum;
}

#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a,
                                          __global float *b, 
                                          __global float *c, 
                                          const int m, 
                                          const int k, 
                                          const int n)
{
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1) * WORK_PER_THREAD;
    const unsigned int local_j = get_local_id(0);
    const unsigned int local_i = get_local_id(1) * WORK_PER_THREAD;

    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (int p = 0; p < WORK_PER_THREAD; p++) {
        sum[p] = 0;
    }
    for (int t = 0; t * TILE_SIZE < k; t++) {
        for (int p = 0; p < WORK_PER_THREAD; p++) {
            a_tile[local_i + p][local_j] = a[(i + p) * k + (t * TILE_SIZE + local_j)];
            b_tile[local_i + p][local_j] = b[(t * TILE_SIZE + local_i + p) * n + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            const float tmp = b_tile[k][local_j];
            for (int r = 0; r < WORK_PER_THREAD; r++) {
                sum[r] += a_tile[local_i + r][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int p = 0; p < WORK_PER_THREAD; p++) {
        c[(i + p) * n + j] = sum[p];
    }
}
#endif
