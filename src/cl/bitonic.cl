__kernel void bitonic(
    __global int *as,
    unsigned int block_size,
    unsigned int swap_len,
    unsigned int n
) {
    unsigned int i = get_global_id(0);
    int j = i + swap_len;
    if (j >= n || (i / swap_len) % 2 == 1)
        return;

    bool block_is_even = (i / block_size) % 2 == 0;
    if ((block_is_even && as[i] > as[j]) || (!block_is_even && as[i] < as[j])) {
        int t = as[i];
        as[i] = as[j];
        as[j] = t;
    }
}
