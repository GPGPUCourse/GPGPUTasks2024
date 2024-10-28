__kernel void naive_prefix_sum(
    const __global unsigned int *data,
    unsigned int n,
    unsigned int block_size,
    __global unsigned int *out
)
{
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    if (i < block_size) {
        out[i] = data[i];
    } else {
        out[i] = data[i] + data[i - block_size];
    }
}

