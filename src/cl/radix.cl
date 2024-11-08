#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define nbits 4

unsigned int get_elem_part(unsigned int val, unsigned int bit_shift) {
    return (val >> bit_shift) % (1 << nbits);
}

__kernel void write_zeros(__global unsigned int *counters) {
    counters[get_global_id(0)] = 0;
    return;
}

__kernel void count_by_wg(__global unsigned int *as, unsigned int n, unsigned int bit_shift) {
    __local unsigned int counters[1 << nbits];
    for (int i = 0; i < 1 << nbits; i++) {
        counters[i] = 0;
    }

    int cur = get_elem_part(as[get_global_id(0)], bit_shift);

    atomic_add(&counters[cur], 1);
    return;
}

#define TILE_SIZE 8
__kernel void matrix_transpose(
        __global float *a,
        __global float *at,
        unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * (1 << nbits) + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_i = i - local_i;
    int tile_j = j - local_j;

    at[(tile_i + local_j) * k + (tile_j + local_i)] = tile[local_i][local_j];
}

__kernel void prefix_stage1(__global unsigned int *as, unsigned int step, unsigned int n) {
    int global_id = get_global_id(0);

    int arr_id = (global_id + 1) * (1 << step) - 1;

    if (arr_id >= n)
        return;

    as[arr_id] += as[arr_id - (1 << (step - 1))];
}

__kernel void prefix_stage2(__global unsigned int *as, unsigned int step, unsigned int n) {
    int global_id = get_global_id(0);

    int cur_block = (1 << step);

    int arr_id = (global_id + 1) * (1 << step) - 1;

    if (arr_id + cur_block / 2 >= n)
        return;

    as[arr_id + cur_block / 2] += as[arr_id];
}

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *prefix_sums, int bit_shift) {
    int gid = get_global_id(0);

    int prev_group_id = gid / get_local_size(0) - 1;

    int cur_val = get_elem_part(as[gid], bit_shift);

    int cur_elem_offset = 0;

    int prev_group_id_offset = cur_val * get_num_groups(0) + prev_group_id;
    if (prev_group_id_offset >= 0) {
        cur_elem_offset += prefix_sums[prev_group_id_offset];
    }

    for (int i = gid - get_local_id(0); i < gid; i++) {
        if (get_elem_part(as[i], bit_shift) == cur_val) {
            ++cur_elem_offset;
        }
    }

    bs[cur_elem_offset] = cur_val;
}