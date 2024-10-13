#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

int imax(int a, int b) {
    return a > b ? a : b;
}

int imin(int a, int b) {
    return a < b ? a : b;
}

int calculate_diagonal(const int *a, const int *b, int N, int M, int i) {
    int l = imax(0, i - M), r = imin(N, i) + 1;
    while (r - l > 1) {
        int m = (l + r) / 2;
        if (a[m - 1] > b[i - m]) {
            r = m;
        } else {
            l = m;
        }
    }
    return l;
}

void diagonal_merge(const int *a, const int *b, int *res, int N, int M, int i) {
    int l = calculate_diagonal(a, b, N, M, i);

    if (l == N) {
        res[i] = b[i - N];
    } else if (i - l == M) {
        res[i] = a[l];
    } else {
        res[i] = a[l] <= b[i - l] ? a[l] : b[i - l];
    }
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    int N = block_size;
    int two_block_size = N * 2;
    int global_index = get_global_id(0);
    int block_i = global_index / two_block_size;
    int a_block_offset = block_i * two_block_size;
    int i = global_index - a_block_offset;

    const int *a = as + a_block_offset;
    const int *b = a + N;
    int *res = bs + a_block_offset;

    diagonal_merge(a, b, res, N, N, i);
}

#define WORKGROUP_SIZE 128

// Это чтобы не думать о том, что WORKGROUP_SIZE > block_size в 3.2
__kernel void merge_sort_small(__global int *as) {
    int global_index = get_global_id(0);
    int i = get_local_id(0);

    __local int a_local[WORKGROUP_SIZE];
    __local int b_local[WORKGROUP_SIZE];
    int *a = a_local;
    int *b = b_local;

    a[i] = as[global_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    int block_size = 1, two_block_size = 2;
    while (block_size < WORKGROUP_SIZE) {
        int block_start_offset = i & ~(two_block_size - 1);
        int bi = i - block_start_offset;
        diagonal_merge(a + block_start_offset, a + block_start_offset + block_size, b + block_start_offset, block_size, block_size, bi);
        block_size = two_block_size;
        two_block_size *= 2;
        int *temp = a;
        a = b;
        b = temp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as[global_index] = a[i];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{
    int N = block_size;
    int two_block_size = N + N;
    int global_index = get_global_id(0);
    int i = global_index * WORKGROUP_SIZE;
    int block_offset = i & ~(two_block_size - 1);
    int bi = i - block_offset;

    const int *a = as + block_offset;
    const int *b = a + N;

    int l = calculate_diagonal(a, b, N, N, bi);
    inds[global_index] = l;
}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size, int chunk_count)
{
    int N = block_size;
    int two_block_size = N + N;
    int global_index = get_global_id(0);
    int block_offset = global_index & ~(two_block_size - 1);
    int chunk_offset = get_group_id(0) * WORKGROUP_SIZE;
    int chunk_index = get_local_id(0);
    int chunk_id = get_group_id(0);
    int chunks_per_block = two_block_size / WORKGROUP_SIZE;

    int l0 = inds[chunk_id];
    int l1 = (chunk_id & (chunks_per_block - 1)) != chunks_per_block - 1 ? inds[chunk_id + 1] : block_size;

    int a_len = l1 - l0;
    const int *a = as + block_offset + l0;
    const int *b = as + block_offset + N + (chunk_offset - block_offset - l0);

    __local int mem[WORKGROUP_SIZE];

    if (chunk_index < a_len) {
        mem[chunk_index] = a[chunk_index];
    } else {
        mem[chunk_index] = b[chunk_index - a_len];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    diagonal_merge(mem, mem + a_len, bs + chunk_offset, a_len, WORKGROUP_SIZE - a_len, chunk_index);
}
