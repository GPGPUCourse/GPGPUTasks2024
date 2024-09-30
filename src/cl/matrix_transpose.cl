#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float *as, __global float *as_t, unsigned int M, unsigned int K)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    as_t[M * i + j] = as[K * j + i];
}

__kernel void matrix_transpose_local_bad_banks()
{
    // TODO
}

__kernel void matrix_transpose_local_good_banks()
{
    // TODO
}
