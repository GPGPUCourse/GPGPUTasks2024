#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += a[k + j * K] * b[i + k * N];
    }

    c[i + j * N] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (int tile_k = 0; tile_k * TILE_SIZE < K; tile_k++) {
        int tile_i = local_i + tile_k * TILE_SIZE;
        int tile_j = local_j + tile_k * TILE_SIZE;

        tile_a[local_j][local_i] = a[tile_i + j * K];
        tile_b[local_j][local_i] = b[i + tile_j * N];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_j][k] * tile_b[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[i + j * N] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
#define RTS (TILE_SIZE / WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_group_id(0) * TILE_SIZE + local_i;
    int j = get_group_id(1) * TILE_SIZE + local_j;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float acc[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        acc[w] = 0;
    }

    for (int t = 0; t < K / TILE_SIZE; t++) {
        for (int w = 0; w < WORK_PER_THREAD; w++) {
            tile_a[local_j + w * RTS][local_i] = a[(local_i + t * TILE_SIZE) + (j       + w * RTS)                 * K];
            tile_b[local_j + w * RTS][local_i] = b[i                         + (local_j + t * TILE_SIZE + w * RTS) * N];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            for (int w = 0; w < WORK_PER_THREAD; w++) {
                acc[w] += tile_a[local_j + w * RTS][k] * tile_b[k][local_i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        c[(j + w * RTS) * N + i] = acc[w];
    }
}
#endif

