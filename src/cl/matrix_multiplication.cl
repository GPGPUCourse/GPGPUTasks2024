#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_i][local_j] = a[i * K + local_j + tileK * TILE_SIZE];
        tileB[local_i][local_j] = b[(local_i + tileK * TILE_SIZE) * N + j];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[local_i][i] * tileB[i][local_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[i * N + j] = sum;
   }
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N) {
     int i = get_global_id(0);
     int j = get_global_id(1);
     int local_i = get_local_id(0);
     int local_j = get_local_id(1);

     __local float tileA[TILE_SIZE][TILE_SIZE];
     __local float tileB[TILE_SIZE][TILE_SIZE];

     float sum[WORK_PER_THREAD];
     for (int i = 0; i < WORK_PER_THREAD; i++) {
        sum[i] = 0;
     }

     for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
         for (int thread = 0; thread < WORK_PER_THREAD; thread++) {
             tileA[local_i * WORK_PER_THREAD + thread][local_j] = a[(i * WORK_PER_THREAD + thread) * K + local_j + tileK * TILE_SIZE];
             tileB[local_i * WORK_PER_THREAD + thread][local_j] = b[((local_i * WORK_PER_THREAD + thread) + tileK * TILE_SIZE) * N + j];
         }
         barrier(CLK_LOCAL_MEM_FENCE);

         for (int i = 0; i < TILE_SIZE; i++) {
            float tileb = tileB[i][local_j];
            for (int thread = 0; thread < WORK_PER_THREAD; thread++) {
                sum[thread] += tileA[local_i * WORK_PER_THREAD + thread][i] * tileb;
             }
         }
         barrier(CLK_LOCAL_MEM_FENCE);
     }

   for (int thread = 0; thread < WORK_PER_THREAD; thread++) {
      c[(i * WORK_PER_THREAD + thread) * N + j] = sum[thread];
   }
}
#endif
