__kernel void write_zeros(__global unsigned int* buf, unsigned int n) {
    const size_t i = get_global_id(0);
    if (i < n) {
        buf[i] = 0;
    }
}

#define TILE_SIZE 16
__kernel void transpose(
    __global const float* in,
    __global float* out,
    unsigned int M,
    unsigned int K
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= K || j >= M) {
        return;
    }

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float local_buffer[TILE_SIZE][TILE_SIZE + 1];

    local_buffer[local_j][local_i] = in[j * K + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int target_i = i - local_i + local_j;
    const unsigned int target_j = j - local_j + local_i;

    out[target_i * M + target_j] = local_buffer[local_i][local_j];
}
#undef TILE_SIZE

__kernel void prefix_sum_pass1(
    __global unsigned int* as,
    unsigned int as_size,
    unsigned int block_size
) {
    unsigned int i = get_global_id(0);
    unsigned int last_in_block = (1 + i) * block_size - 1;
    unsigned int mid_in_block = last_in_block - block_size / 2;
    if (last_in_block < as_size) {
        as[last_in_block] = as[mid_in_block] + as[last_in_block];
    }
}

__kernel void prefix_sum_pass2(
    __global unsigned int* as,
    unsigned int as_size,
    unsigned int block_size
) {
    unsigned int i = get_global_id(0);
    unsigned int target_i = (1 + i) * block_size * 2 + block_size - 1;
    if (target_i < as_size) {
        as[target_i] = as[target_i - block_size] + as[target_i];
    }
}

#define NBITS 4
__kernel void count(
    __global const unsigned int* as,
    unsigned int n,
    __global unsigned int* buf,
    unsigned int bit_shift
) {
    const size_t i = get_global_id(0);
    if (i < n) {
        const size_t wg_id = i / get_local_size(0);
        const unsigned int ndigits = 1 << NBITS;
        const unsigned int d = (as[i] >> bit_shift) & (ndigits - 1);
        atomic_inc(&buf[wg_id * ndigits + d]);
    }
}

__kernel void sort(
    __global const unsigned int* as,
    __global unsigned int* bs,
    unsigned int n,
    __global const unsigned int* buf,
    unsigned int bit_shift
) {
    const size_t i = get_global_id(0);
    if (i < n) {
        const size_t nwg = get_global_size(0) / get_local_size(0);
        const size_t wg_id = i / get_local_size(0);
        const unsigned int ndigits = 1 << NBITS;
        const unsigned int d = (as[i] >> bit_shift) & (ndigits - 1);
        unsigned int count_i = d * nwg + wg_id;
        unsigned int place = count_i > 0 ? buf[count_i - 1] : 0;

        for (unsigned int k = wg_id * get_local_size(0); k < i; ++k) {
            place += ((as[k] >> bit_shift) & (ndigits - 1)) == d;
        }

        bs[place] = as[i];
    }
}
#undef NBITS
