#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global const float *lhs, 
    __global const float *rhs,
    __global float *result,
    unsigned m, unsigned k, unsigned n)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);

    float sum = 0;
    for (int i = 0; i < k; i++) {
        sum += lhs[gid_x * k + i] * rhs[i * n + gid_y];
    }

    result[gid_x * n + gid_y] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global const float *lhs, 
    __global const float *rhs,
    __global float *result,
    unsigned m, unsigned k, unsigned n)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);

    __local float lhs_tile[TILE_SIZE][TILE_SIZE];
    __local float rhs_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (int tile = 0; tile < k; tile += TILE_SIZE) {
        lhs_tile[lid_x][lid_y] = lhs[gid_x * k + lid_y + tile];
        rhs_tile[lid_x][lid_y] = rhs[(lid_x + tile) * n + gid_y];
    
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += lhs_tile[lid_x][i] * rhs_tile[i][lid_y];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[gid_x * n + gid_y] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global const float *lhs, 
    __global const float *rhs,
    __global float *result,
    unsigned m, unsigned k, unsigned n)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);

    __local float lhs_tile[TILE_SIZE][TILE_SIZE];
    __local float rhs_tile[TILE_SIZE][TILE_SIZE];

    // for some reason, `sum[WORK_PER_THREAD] = { 0 }` doesn't work (sic!):
    float sum[WORK_PER_THREAD];
    for (int i = 0; i < WORK_PER_THREAD; i++) {
        sum[i] = 0;
    }
    for (int tile = 0; tile < k; tile += TILE_SIZE) {
        for (int i = 0; i < WORK_PER_THREAD; i++) {
            lhs_tile[lid_x * WORK_PER_THREAD + i][lid_y] = lhs[(gid_x * WORK_PER_THREAD + i) * k + lid_y + tile];
            rhs_tile[lid_x * WORK_PER_THREAD + i][lid_y] = rhs[(lid_x * WORK_PER_THREAD + tile + i) * n + gid_y];
        }
    
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < WORK_PER_THREAD; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                sum[i] += lhs_tile[lid_x * WORK_PER_THREAD + i][j] * rhs_tile[j][lid_y];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < WORK_PER_THREAD; i++) {
        result[(gid_x * WORK_PER_THREAD + i) * n + gid_y] = sum[i];
    }
}
#endif
