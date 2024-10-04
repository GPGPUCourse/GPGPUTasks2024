#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
        __global float *as,
        __global float *bs,
        __global float *cs,
        unsigned int m,
        unsigned int k,
        unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    float sum = 0;
    for (unsigned int l = 0; l < k; l++) {
        sum += as[j * k + l] * bs[l * n + i];
    }

    cs[j * n + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
        __global float *as,
        __global float *bs,
        __global float *cs,
        unsigned int m,
        unsigned int k,
        unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int li = get_local_id(0);
    unsigned int lj = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (int base_k = 0; base_k < k; base_k += TILE_SIZE) {
        tile_a[lj][li] = as[j * k + base_k + li];
        tile_b[lj][li] = bs[(base_k + lj) * n + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; l++) {
            sum += tile_a[lj][l] * tile_b[l][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[j * n + i] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(
        __global float *as,
        __global float *bs,
        __global float *cs,
        unsigned int m,
        unsigned int k,
        unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int li = get_local_id(0);
    unsigned int lj = get_local_id(1);
    unsigned int gi = get_group_id(0);
    unsigned int gj = get_group_id(1);
    const int WPT = WORK_PER_THREAD;
    const int TPT = TILE_SIZE / WPT;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum[WPT];
    for (int w = 0; w < WPT; w++) {
        sum[w] = 0;
    }

    for (int base_k = 0; base_k < k; base_k += TILE_SIZE) {
        for (int w = 0; w < WPT; w++) {
            unsigned int wlj = w * TPT + lj;
            unsigned int wj = wlj + gj * TILE_SIZE;
            tile_a[wlj][li] = as[wj * k + base_k + li];
            tile_b[wlj][li] = bs[(base_k + wlj) * n + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; l++) {
            for (int w = 0; w < WPT; w++) {
                unsigned int wlj = w * TPT + lj;
                sum[w] += tile_a[wlj][l] * tile_b[l][li];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; w++) {
        unsigned int wlj = w * TPT + lj;
        cs[(gj * TILE_SIZE + wlj) * n + i] = sum[w];
    }
}
#endif
