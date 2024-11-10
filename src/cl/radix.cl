#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define nbits 4

unsigned int get_elem_part(unsigned int val, unsigned int bit_shift) {
    return (val >> bit_shift) % (1 << nbits);
}

__kernel void count_by_wg(__global unsigned int *as, __global unsigned int *g_counters, unsigned int bit_shift) {
    __local unsigned int counters[1 << nbits];

    int cur = get_elem_part(as[get_global_id(0)], bit_shift);

    atomic_add(&counters[cur], 1);

    if (get_local_id(0) == 0) {
        for (int i = 0; i < 1 << nbits; i++) {
            g_counters[i * get_num_groups(0) + get_group_id(0)] = counters[i];
        }
    }
    return;
}

__kernel void matrix_transpose(
        __global float *a,
        __global float *at,
        unsigned int m
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    at[j * (1 << nbits) + i] = a[i * m + j];
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

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *prefix_sums, int bit_shift, unsigned int n) {
    int gid = get_global_id(0);

    int prev_group_id = gid / get_local_size(0) - 1;

    int cur_val = get_elem_part(as[gid], bit_shift);

    int cur_elem_offset = 0;

    int prev_group_id_offset = cur_val * get_num_groups(0) + prev_group_id;
    if (prev_group_id_offset >= 0) {
        cur_elem_offset += prefix_sums[prev_group_id_offset];
    }

    for (int i = gid - get_local_id(0); i < gid; i++) { // todo: preliminary sort is needed
        if (get_elem_part(as[i], bit_shift) == cur_val) {
            ++cur_elem_offset;
        }
    }

    bs[cur_elem_offset] = as[gid];
}