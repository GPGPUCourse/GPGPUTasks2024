#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16
#line 6

__kernel void matrix_transpose_naive(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= k || j >= m)
        return;

    at[j * m + i] = a[i * k + j];
}

__kernel void matrix_transpose_local_bad_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0); // Номер столбца в A
    int j = get_global_id(1); // Номер строчки в A

    __local float tile[TILE_SIZE * TILE_SIZE];

    int i_local = get_local_id(0);  // Номер столбца в tile
    int j_local = get_local_id(1);  // Номер строчки в tile

    tile[i_local * TILE_SIZE + j_local] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    at[j * m + i] = tile[i_local * TILE_SIZE + j_local];
}

__kernel void matrix_transpose_local_good_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0); // Номер столбца в A
    int j = get_global_id(1); // Номер строчки в A

    __local float tile[TILE_SIZE * (TILE_SIZE + 1)];

    int i_local = get_local_id(0);  // Номер столбца в tile
    int j_local = get_local_id(1);  // Номер строчки в tile

    tile[i_local * (TILE_SIZE) + j_local] = a[i * k + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    at[j * m + i] = tile[i_local * (TILE_SIZE) + j_local];
}
