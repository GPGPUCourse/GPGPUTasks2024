#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(
    __global float* a,
    __global float* a_t,
    unsigned int M, unsigned int K
)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (j >= M || i >= K) {
        return;
    }

    float x = a[j * K + i];
    a_t[i * M + j] = x;
}

__kernel void matrix_transpose_local_bad_banks(
    __global float* a,
    __global float* a_t,
    unsigned int M, unsigned int K
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y >= M || x >= K) {
        return;
    }

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    tile[local_y][local_x] = a[y * K + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_y_T = get_group_id(1) * TILE_SIZE;
    int tile_x_T = get_group_id(0) * TILE_SIZE;

    int in_tile_x_T = local_y;
    int in_tile_y_T = local_x;

    a_t[(tile_x_T + in_tile_x_T) * M + (tile_y_T + in_tile_y_T)] = tile[local_x][local_y];
}

__kernel void matrix_transpose_local_good_banks(
    __global float* a,
    __global float* a_t,
    unsigned int M, unsigned int K
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y >= M || x >= K) {
        return;
    }

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    tile[local_y][local_x] = a[y * K + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_y_T = get_group_id(1) * TILE_SIZE;
    int tile_x_T = get_group_id(0) * TILE_SIZE;

    int in_tile_x_T = local_y;
    int in_tile_y_T = local_x;

    a_t[(tile_x_T + in_tile_x_T) * M + (tile_y_T + in_tile_y_T)] = tile[local_x][local_y];
}
