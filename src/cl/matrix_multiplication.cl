#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
        __global const float *a, __global const float *b, __global float *c,
        int m, int k, int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float s = 0;
    for (int l = 0; l < k; l++) {
        s += a[j * k + l] * b[l * n + i];
    }

    c[j * n + i] = s;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
        __global const float *a, __global const float *b, __global float *c,
        int m, int k, int n)
{
    int col = get_local_id(0);
    int row = get_local_id(1);

    int global_col = get_group_id(0) * TILE_SIZE + col;
    int global_row = get_group_id(1) * TILE_SIZE + row;

    __local float TileA[TILE_SIZE][TILE_SIZE];
    __local float TileB[TILE_SIZE][TILE_SIZE];

    float s = 0;
    for (int t = 0; t * TILE_SIZE < k; t++) {
        TileA[row][col] = a[global_row * k + t * TILE_SIZE + col];
        TileB[row][col] = b[(t * TILE_SIZE + row) * n + global_col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            s += TileA[row][i] * TileB[i][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_row * n + global_col] = s;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
        __global const float *a, __global const float *b, __global float *c,
        int m, int k, int n)
{
    int col = get_local_id(0);
    int row = get_local_id(1);

    const int RTS = TILE_SIZE / WORK_PER_THREAD;

    int global_col = get_group_id(0) * TILE_SIZE + col;
    int global_row = get_group_id(1) * TILE_SIZE + row;

    __local float TileA[TILE_SIZE][TILE_SIZE];
    __local float TileB[TILE_SIZE][TILE_SIZE];

    float s[WORK_PER_THREAD] = {0};

    for (int t = 0; t * TILE_SIZE < k; t++) {
        for (int w = 0; w < WORK_PER_THREAD; w++) {
            TileA[row + w * RTS][col] = a[(global_row + w * RTS) * k + t * TILE_SIZE + col];
            TileB[row + w * RTS][col] = b[(t * TILE_SIZE + row + w * RTS) * n + global_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            for (int w = 0; w < WORK_PER_THREAD; w++) {
                s[w] += TileA[row + w * RTS][i] * TileB[i][col];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        c[(global_row + w * RTS) * n + global_col] = s[w];
    }
}
#endif
