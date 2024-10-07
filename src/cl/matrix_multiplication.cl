#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global float *a, __global float *b, __global float *c,
    unsigned int m, unsigned int k, unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= n || j >= m) {
        return;
    }

    c[j * n + i] = 0;

    for (unsigned int t = 0; t < k; t++) {
        c[j * n + i] += a[j * k + t] * b[t * n + i];
    }
}

#ifdef TILE_SIZE

__kernel void matrix_multiplication_local(
    __global float *a, __global float *b, __global float *c,
    unsigned int m, unsigned int k, unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (unsigned int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (i < m && (t * TILE_SIZE + local_j) < k) {
            tile_a[local_i][local_j] = a[k * i + (t * TILE_SIZE + local_j)];
        } else {
            tile_a[local_i][local_j] = 0.f;
        }

        if ((t * TILE_SIZE + local_i) < k && j < n) {
            tile_b[local_i][local_j] = b[n * (t * TILE_SIZE + local_i) + j];
        } else {
            tile_b[local_i][local_j] = 0.f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_i][k] * tile_b[k][local_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < m && j < n) {
        c[n * i + j] = sum;
    }
}

#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)

#define TILE_PARTS TILE_SIZE / WORK_PER_THREAD

__kernel void matrix_multiplication_local_wpt(
    __global float *a, __global float *b, __global float *c,
    unsigned int m, unsigned int k, unsigned int n
) {
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int i = TILE_SIZE * get_group_id(0) + local_i;
    unsigned int j = TILE_SIZE * get_group_id(1) + local_j;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float res[WORK_PER_THREAD];
    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        res[w] = 0.f;
    }

    for (unsigned int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
            if (i < m && t * TILE_SIZE + local_j + w * TILE_PARTS < k) {
                tile_a[local_i][local_j + TILE_PARTS * w] = a[i * k + t * TILE_SIZE + local_j + w * TILE_PARTS];
            } else {
                tile_a[local_i][local_j + TILE_PARTS * w] = 0.f;
            }
            if (t * TILE_SIZE + local_i < k && j + w * TILE_PARTS < n) {
                tile_b[local_i][local_j + w * TILE_PARTS] = b[n * (t * TILE_SIZE + local_i) + j + w * TILE_PARTS];
            } else {
                tile_b[local_i][local_j] = 0.f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            float tile_a_value = tile_a[local_i][k];

            for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
                res[w] += tile_a_value * tile_b[k][local_j + w * TILE_PARTS];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        c[n * i + j + w * TILE_PARTS] = res[w];
    }
}

#endif
