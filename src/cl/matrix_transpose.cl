#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global const float *as, __global float *as_t, unsigned int m, unsigned int k) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= k || j >= m) return;

    as_t[i * m + j] = as[j * k + i];
}

__kernel void matrix_transpose_local_bad_banks(__global const float *as, __global float *as_t, unsigned int m, unsigned int k) {
    unsigned int i = LINES_PER_GROUP0 * get_local_id(0);
    unsigned int j = LINES_PER_GROUP1 * get_local_id(1);
    unsigned int wi = TILE_SIZE * get_group_id(0) + i;
    unsigned int wj = TILE_SIZE * get_group_id(1) + j;

    __local float buf[TILE_SIZE][TILE_SIZE];

    for (int dj = 0; dj < LINES_PER_GROUP1; ++dj) {
        for (int di = 0; di < LINES_PER_GROUP0; ++di) {
            buf[i + di][j + dj] = wi + di >= k || wj + dj >= m ? 0 : as[(wj + dj) * k + wi + di];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (wi >= k || wj >= m) return;
    for (int dj = 0; dj < LINES_PER_GROUP1; ++dj) {
        for (int di = 0; di < LINES_PER_GROUP0; ++di) {
            as_t[(wi - i + j + dj) * m + (wj - j + i + di)] = buf[j + dj][i + di];
        }
    }
}

__kernel void matrix_transpose_local_good_banks(__global const float *as, __global float *as_t, unsigned int m, unsigned int k) {
    unsigned int i = get_local_id(0);
    unsigned int j = get_local_id(1);
    unsigned int wi = TILE_SIZE * get_group_id(0) + i;
    unsigned int wj = TILE_SIZE * get_group_id(1) + j;

    __local float buf[TILE_SIZE][TILE_SIZE + 1];

    const int stride_j = TILE_SIZE / LINES_PER_GROUP1;
    const int stride_i = TILE_SIZE / LINES_PER_GROUP0;

    for (int dj = 0; dj < LINES_PER_GROUP1; ++dj) {
        for (int di = 0; di < LINES_PER_GROUP0; ++di) {
            buf[i + di * stride_i][j + dj * stride_j] = wi + di * stride_i >= k || wj + dj * stride_j >= m ? 0 : as[(wj + dj * stride_j) * k + wi + di * stride_i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (wi >= k || wj >= m) return;
    for (int dj = 0; dj < LINES_PER_GROUP1; ++dj) {
        for (int di = 0; di < LINES_PER_GROUP0; ++di) {
            as_t[(wi - i + j + dj * stride_j) * m + (wj - j + i + di * stride_i)] = buf[j + dj * stride_j][i + di * stride_i];
        }
    }
}
