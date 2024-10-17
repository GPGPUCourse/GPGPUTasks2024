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

    __local float lhs_tile[TILE_SIZE][TILE_SIZE + 1];
    __local float rhs_tile[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0;
    for (int tile = 0; tile < k; tile += TILE_SIZE) {
        lhs_tile[lid_y][lid_x] = lhs[gid_y * k + tile + lid_x];
        rhs_tile[lid_y][lid_x] = rhs[(tile + lid_y) * n + gid_x];
    
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += lhs_tile[lid_y][i] * rhs_tile[i][lid_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[gid_y * n + gid_x] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
    __global const float *lhs, 
    __global const float *rhs,
    __global float *result,
    unsigned m, unsigned k, unsigned n)
{
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int gid_x = get_group_id(0) * TILE_SIZE + lid_x;
    const int gid_y = get_group_id(1) * TILE_SIZE + lid_y;

    __local float lhs_tile[TILE_SIZE][TILE_SIZE + 1];
    __local float rhs_tile[TILE_SIZE][TILE_SIZE + 1];

    // for some reason, `sum[WORK_PER_THREAD] = { 0 }` doesn't work (sic!):
    float sum[WORK_PER_THREAD];
    for (int i = 0; i < WORK_PER_THREAD; i++) {
        sum[i] = 0;
    }

    for (int tile = 0; tile < k; tile += TILE_SIZE) {
        for (int i = 0; i < WORK_PER_THREAD; i++) {
            const int tid = i * (TILE_SIZE / WORK_PER_THREAD) + lid_y;
            const int lhs_tid = (i * (TILE_SIZE / WORK_PER_THREAD) + gid_y) * k + lid_x + tile;
            const int rhs_tid = (i * (TILE_SIZE / WORK_PER_THREAD) + lid_y + tile) * n + gid_x;

            lhs_tile[tid][lid_x] = lhs[lhs_tid];
            rhs_tile[tid][lid_x] = rhs[rhs_tid];
        }
    
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < WORK_PER_THREAD; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                sum[i] += lhs_tile[i * (TILE_SIZE / WORK_PER_THREAD) + lid_y][j] * rhs_tile[j][lid_x];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < WORK_PER_THREAD; i++) {
        result[(i * (TILE_SIZE / WORK_PER_THREAD) + gid_y) * n + gid_x] = sum[i];
    }
}
#endif
