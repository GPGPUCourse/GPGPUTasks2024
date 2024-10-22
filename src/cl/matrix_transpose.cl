#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose_naive(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = a[j * k + i];
    at[i * m + j] = x;
}

__kernel void matrix_transpose_local_bad_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE]; 
    int li = get_local_id(0);
    int lj = get_local_id(1);
    
    tile[lj][li] = a[j * k + i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    at[(i - li + lj) * m + j - lj + li] = tile[li][lj];
}

__kernel void matrix_transpose_local_good_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE + 1]; 
    int li = get_local_id(0);
    int lj = get_local_id(1);
    
    tile[lj][li] = a[j * k + i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    at[(i - li + lj) * m + j - lj + li] = tile[li][lj];
}
