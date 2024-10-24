#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a, 
                                        __global float *b, 
                                        __global float *c, 
                                        unsigned int M, 
                                        unsigned int K, 
                                        unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a, 
                                        __global float *b, 
                                        __global float *c, 
                                        unsigned int M, 
                                        unsigned int K, 
                                        unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int li = get_local_id(0);
    int lj = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t * TILE_SIZE < K; t++) {
        tileA[lj][li] = a[j * K + (t * TILE_SIZE + li)];
        tileB[lj][li] = b[(t * TILE_SIZE + lj) * N + i];
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[lj][k] * tileB[k][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    c[j * N + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a, 
                                        __global float *b, 
                                        __global float *c, 
                                        unsigned int M, 
                                        unsigned int K, 
                                        unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int li = get_local_id(0);
    int lj = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        sum[w] = 0;
    }

    for (int t = 0; t * TILE_SIZE < K; t++) {
        for (int w = 0; w < WORK_PER_THREAD; w++) {
            tileA[lj * WORK_PER_THREAD + w][li] = a[(j * WORK_PER_THREAD + w) * K + (t * TILE_SIZE + li)];
            tileB[lj * WORK_PER_THREAD + w][li] = b[(t * TILE_SIZE + (lj * WORK_PER_THREAD + w)) * N + i];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int w = 0; w < WORK_PER_THREAD; w++) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum[w] += tileA[lj * WORK_PER_THREAD + w][k] * tileB[k][li];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for(int w = 0; w < WORK_PER_THREAD; w++) {
        c[(j * WORK_PER_THREAD + w) * N + i] = sum[w];
    }
}
#endif
