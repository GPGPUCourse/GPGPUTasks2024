#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16
#line 6

__kernel void matrix_transpose_naive(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= k || j >= m)
        return;

    at[j * m + i] = a[i * k + j];
}

__kernel void matrix_transpose_local_bad_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    unsigned int i = get_global_id(0); // Номер столбца в A
    unsigned int j = get_global_id(1); // Номер строчки в A

    unsigned int i_local = get_local_id(0);  // Номер столбца в tile
    unsigned int j_local = get_local_id(1);  // Номер строчки в tile

    __local float tile[TILE_SIZE * TILE_SIZE];
    tile[j_local * TILE_SIZE + i_local] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i_group = get_group_id(0);
    unsigned int j_group = get_group_id(1);

    at[(i_group * TILE_SIZE + j_local) * m + (j_group * TILE_SIZE + i_local)] = tile[i_local * TILE_SIZE + j_local];
}

__kernel void matrix_transpose_local_good_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    unsigned int i = get_global_id(0); // Номер столбца в A
    unsigned int j = get_global_id(1); // Номер строчки в A

    __local float tile[TILE_SIZE * (TILE_SIZE + 1)];

    unsigned int i_local = get_local_id(0);  // Номер столбца в tile
    unsigned int j_local = get_local_id(1);  // Номер строчки в tile

    tile[j_local * TILE_SIZE + i_local] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i_group = get_group_id(0);
    unsigned int j_group = get_group_id(1);

    at[(i_group * TILE_SIZE + j_local) * m + (j_group * TILE_SIZE + i_local)] = tile[i_local * TILE_SIZE + j_local];
}
