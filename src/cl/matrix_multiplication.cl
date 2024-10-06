#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

// I work in vstudio
// #line 6 

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global const float* a,
    __global const float* b,
    __global float* c,
    unsigned int M, 
    unsigned int K,
    unsigned int N
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += a[y * K + k] * b[k * N + x];
    }
    c[y * N + x] = acc;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global const float* a,
    __global const float* b,
    __global float* c,
    unsigned int M, 
    unsigned int K,
    unsigned int N
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const int numTiles = K / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int x_a = t * TILE_SIZE + local_x;
        int y_a = y;

        int x_b = x;
        int y_b = t * TILE_SIZE + local_y;

        tile_a[local_y][local_x] = a[y_a * K + x_a];
        tile_b[local_y][local_x] = b[y_b * N + x_b];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_y][k] * tile_b[k][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * N + x] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global const float* a,
    __global const float* b,
    __global float* c,
    unsigned int M, 
    unsigned int K,
    unsigned int N
)
{    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int x = get_group_id(0) * TILE_SIZE + local_x;
    int y = get_group_id(1) * TILE_SIZE + local_y;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float reg[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        reg[w] = 0.0f;
    }

    const int numTiles = K / TILE_SIZE;
    const int RTS = TILE_SIZE / WORK_PER_THREAD;

    for (int t = 0; t < numTiles; t++) {
        for (int w = 0; w < WORK_PER_THREAD; w++) {
            int x_a = t * TILE_SIZE + local_x;
            int y_a = y + w * RTS;
        
            int x_b = x;
            int y_b = t * TILE_SIZE + local_y + w * RTS;
            
            tile_a[local_y + w * RTS][local_x] = a[y_a * K + x_a];
            tile_b[local_y + w * RTS][local_x] = b[y_b * N + x_b];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            for (int w = 0; w < WORK_PER_THREAD; w++) {
                reg[w] += tile_a[local_y + w * RTS][k] * tile_b[k][local_x];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        c[(y + w * RTS) * N + x] = reg[w];
    }
}
#endif
