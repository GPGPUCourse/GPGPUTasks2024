#define BITS 4
#define WG_SIZE 128
#define TILE_SIZE 16

int get_bits(unsigned int a, unsigned int shift) {
    return (a >> shift) & ((1 << BITS) - 1);
}

__kernel void assign_zeros(__global unsigned int* a)
{
    int gid = get_global_id(0);
    a[gid] = 0;
}

__kernel void count_workgroup(
    __global const unsigned int* a, 
    __global unsigned int* cnt,
    unsigned int shift)
{
    int gid = get_global_id(0);
    int wid = get_group_id(0);
    atomic_inc(&cnt[wid * (1 << BITS) + get_bits(a[gid], shift)]);
}

__kernel void reduce(__global const unsigned int* a, __global unsigned int* b) {
    int i = get_global_id(0);

    b[i] = a[2 * i] + a[2 * i + 1];
}

__kernel void down_sweep(__global const unsigned int* a, __global unsigned int* b) {
    int i = get_global_id(0);

    if (i == 0) {
        return;
    }

    if (i % 2) {
        b[i] = a[i / 2];
    } else {
        b[i] += a[i / 2 - 1];
    }
}

__kernel void matrix_transpose(__global const float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i_res = get_group_id(0) * TILE_SIZE + local_j;
    int j_res = get_group_id(1) * TILE_SIZE + local_i;
    at[i_res * m + j_res] = tile[local_i][local_j];
}

__kernel void radix_sort(
    __global const unsigned int* a,
    __global unsigned int* b, 
    __global unsigned int* cnt,
    unsigned int wg_cnt,
    unsigned int shift)
{
    int lid = get_local_id(0);
    int wid = get_group_id(0);
    int gid = get_global_id(0);

    int num = get_bits(a[gid], shift);
    int cnt_idx = num * wg_cnt + wid - 1;
    int idx = cnt_idx < 0 ? 0 : cnt[cnt_idx];

    for (int i = wid * WG_SIZE; i < wid * WG_SIZE + lid; i++) {
        idx += get_bits(a[i], shift) == num;
    }

    b[idx] = a[gid];
}