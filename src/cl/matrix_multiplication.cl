#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float* a, __global float* b, __global float* c, const int m, const int k, const int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= m || j >= n) return;

    float sum = 0;
    for (int p = 0; p < k; p++) {
        sum += a[i * k + p] * b[p * n + j];
    }
    c[i * n + j] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float* a, __global float* b, __global float* c, const int m, const int k, const int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int i_local = get_local_id(0);
    int j_local = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    if (i >= m || j >= n) return;

    float sum = 0;
    for (int tileK = 0; tileK * TILE_SIZE < k; tileK++) {
        int global_tile_i = tileK * TILE_SIZE + i_local;
        int global_tile_j = tileK * TILE_SIZE + j_local;

        if (global_tile_i < m && global_tile_j < k) {
            tileA[i_local][j_local] = a[i * k + global_tile_j];
        }

        if (global_tile_j < k && global_tile_i < n) {
            tileB[i_local][j_local] = b[global_tile_i * n + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TILE_SIZE; p++) {
            sum += tileA[j_local * TILE_SIZE][p] * tileB[p][i_local];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[i * n + j] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt()
{
    // TODO
}
#endif
