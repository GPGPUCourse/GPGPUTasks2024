__kernel void zero_memory(__global unsigned int *a)
{
    unsigned int global_id = get_global_id(0);
    a[global_id] = 0;
}

#define TILE_SIZE 16
__kernel void transpose(__global unsigned int *a, __global unsigned int *a_t, unsigned int m, unsigned int k)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local unsigned int buf[TILE_SIZE][TILE_SIZE + 1];

    unsigned int idx = (local_i + local_j) % TILE_SIZE;
    if (i < m && j < k) {
        buf[local_j][idx] = a[j * m + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int out_i = j - local_j + local_i;
    unsigned int out_j = i - local_i + local_j;
    if (out_i < k && out_j < m) {
        a_t[out_i + out_j * k] = buf[local_i][idx];
    }
}

__kernel void calculate_partial_sums(__global unsigned int *partial_sums, unsigned int size)
{
    unsigned int id = get_global_id(0);
    partial_sums[id * size] += partial_sums[id * size + size / 2];
}

__kernel void calculate_prefix_sums(__global unsigned int *result, __global unsigned int *partial_sums, unsigned int size)
{
    unsigned int id = get_global_id(0) + 1;
    if (id & size) {
        result[id] += partial_sums[(id - size) / size * size];
    }
}

#define BITS 4
#define NUM_VALUES (1 << BITS)
unsigned int extract(unsigned int n, unsigned int block_id)
{
    return (n >> (BITS * block_id)) & (NUM_VALUES - 1);
}

__kernel void count(__global unsigned int *a, __global unsigned int *counter, unsigned int block_id)
{
    unsigned int global_id = get_global_id(0);
    unsigned int group_id  = get_group_id(0);
    unsigned int local_id  = get_local_id(0);

    __local unsigned int local_counter[NUM_VALUES];

    if (local_id < NUM_VALUES) {
        local_counter[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_inc(&local_counter[extract(a[global_id], block_id)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < NUM_VALUES) {
        counter[group_id * NUM_VALUES + local_id] = local_counter[local_id];
    }
}

__kernel void radix_sort(__global unsigned int *a, __global unsigned int *prefix_sums, __global unsigned int *out, unsigned int block_id, unsigned int size)
{
    unsigned int global_id = get_global_id(0);
    unsigned int group_id  = get_group_id(0);
    unsigned int local_id  = get_local_id(0);
    unsigned int work_group_size = get_local_size(0);

    unsigned int element_value = extract(a[global_id], block_id);
    unsigned int offset = prefix_sums[element_value * size + group_id];

    for (int i = 0; i < local_id; i++) {
        if (extract(a[group_id * work_group_size + i], block_id) == element_value) {
            offset++;
        }
    }

    out[offset] = a[global_id];
}

