#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = a[j * k + i];
    at[i * m + j] = x;
}

__kernel void matrix_transpose_local_bad_banks()
{
    // TODO
}

__kernel void matrix_transpose_local_good_banks()
{
    // TODO
}