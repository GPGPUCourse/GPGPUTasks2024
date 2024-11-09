#define TILE 16
#define NBITS 4
#define WORK_GROUP_SIZE 128

__kernel void count_by_workgroup(__global unsigned int* as_gpu, __global unsigned int* counters, unsigned int bit_shift, unsigned int n) {
    const unsigned int ndigits = 16;
    __local unsigned int local_counters[ndigits];

    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int local_size = get_local_size(0);

    for (int i = lid; i < ndigits; i += local_size) {
        local_counters[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = gid; i < n; i += get_global_size(0)) {
        unsigned int bucket = (as_gpu[i] >> bit_shift) & (ndigits - 1);
        atomic_add(&local_counters[bucket], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 16) {
        counters[get_group_id(0) * ndigits + lid] = local_counters[lid];
    }
}


__kernel void transpose_counters(__global unsigned int* in, __global unsigned int* out, const unsigned int n, const unsigned int m) {
    unsigned int cols = get_group_id(0) * TILE + get_local_id(0);
    unsigned int rows = get_group_id(1) * TILE + get_local_id(1);

    __local unsigned int tile[TILE][TILE + 1];

    if (cols < m && rows < n) {
        tile[get_local_id(1)][get_local_id(0)] = in[rows * m + cols];
    } else {
        tile[get_local_id(1)][get_local_id(0)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    cols = get_group_id(1) * TILE + get_local_id(0);
    rows = get_group_id(0) * TILE + get_local_id(1);

    if (cols < n && rows < m) {
        out[rows * n + cols] = tile[get_local_id(0)][get_local_id(1)];
    }
}

__kernel void set_zero(__global unsigned int* as_gpu, unsigned int n) {
    if (get_global_id(0) == 0) {
        as_gpu[n - 1] = 0;
    }
}

__kernel void prefix_sum(__global unsigned int* as_gpu, unsigned int offset, unsigned int n, int down) {
    unsigned int gid = get_global_id(0);
    unsigned int index = gid * offset + offset - 1;

    if (index < n) {
        if (down == 0) {
            as_gpu[index] += as_gpu[index - offset / 2];
        } else {
            unsigned int temp = as_gpu[index - offset / 2];
            as_gpu[index - offset / 2] = as_gpu[index];
            as_gpu[index] += temp;
        }
    }
}

__kernel void radix_sort(__global unsigned int* as_gpu, __global unsigned int* bs_gpu, __global unsigned int* cs_gpu, unsigned int shift, unsigned int work_groups) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_id = get_group_id(0);

    unsigned int bits = 31;
    unsigned int digit = (as_gpu[gid] << ((bits - shift) * NBITS)) >> (bits * NBITS);

    __local unsigned int digits[WORK_GROUP_SIZE];
    digits[lid] = digit;
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int offset = 0;
    for (int i = 0; i < lid; i++) {
        if (digits[i] == digit) {
            offset++;
        }
    }

    unsigned int base = cs_gpu[digit * work_groups + group_id];
    bs_gpu[base + offset] = as_gpu[gid];
}