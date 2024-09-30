#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a,
                                          __global float *b,
                                          __global float *c,
                                          unsigned int M,
                                          unsigned int K,
                                          unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    float sum = 0;
    for (unsigned int k = 0; k < K; k++) {
        sum += a[K * j + k] * b[N * k + i];
    }

    c[N * j + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local()
{
    // TODO
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt()
{
    // TODO
}
#endif
