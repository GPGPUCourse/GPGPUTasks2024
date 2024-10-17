#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global float *a, 
                                     __global float *a_t, 
                                     const int m, 
                                     const int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    a_t[i * m + j] = a[j * k + i];
}

#define SIZE 16

__kernel void matrix_transpose_local_bad_banks(__global float *a, 
                                               __global float *a_t, 
                                               const int m, 
                                               const int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    a_t[j * m + i] = a[i * k + j];
}


__kernel void matrix_transpose_local_good_banks(__global float *a, 
                                                __global float *a_t, 
                                                const int m, 
                                                const int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    a_t[j * m + i] = a[i * k + j];
}

