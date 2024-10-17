#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(
    __global const float* matrix, __global float* transposed,
    unsigned int width, unsigned int height)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);

    transposed[gid_x * width + gid_y] = matrix[gid_y * height + gid_x];
}

__kernel void matrix_transpose_local_bad_banks(
    __global const float* matrix, __global float* transposed,
    unsigned int width, unsigned int height)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int tid = (get_group_id(0) * TILE_SIZE + lid_y) * width + get_group_id(1) * TILE_SIZE + lid_x;

    __local float buffer[TILE_SIZE][TILE_SIZE];
    buffer[lid_y][lid_x] = matrix[gid_y * height + gid_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed[tid] = buffer[lid_x][lid_y];
}

__kernel void matrix_transpose_local_good_banks(
    __global const float* matrix, __global float* transposed,
    unsigned int width, unsigned int height)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int tid = (get_group_id(0) * TILE_SIZE + lid_y) * width + get_group_id(1) * TILE_SIZE + lid_x;

    __local float buffer[TILE_SIZE][TILE_SIZE + 1];
    buffer[lid_x][lid_y] = matrix[gid_y * height + gid_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed[tid] = buffer[lid_y][lid_x];
}
