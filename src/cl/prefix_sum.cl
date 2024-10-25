__kernel void naive_sum(
        __global const unsigned int *as,
        __global unsigned int *bs,
        unsigned int sum_len,
        unsigned int n
) {
    unsigned int i = get_global_id(0);

    if (i >= n)
        return;

    if (i < sum_len)
        bs[i] = as[i];
    else
        bs[i] = as[i] + as[i - sum_len];
}

__kernel void up_sweep(
        __global unsigned int *as,
        unsigned int sum_len,
        unsigned int n
) {
    unsigned int i = get_global_id(0);

    if (i * sum_len + sum_len - 1 >= n)
        return;

    as[i * sum_len + sum_len - 1] += as[i * sum_len + sum_len / 2 - 1];
}

__kernel void down_sweep(
        __global unsigned int *as,
        unsigned int sum_len,
        unsigned int n
) {
    // we don't need the first element there
    unsigned int i = get_global_id(0) + 1;

    if (i * sum_len + sum_len / 2 - 1 >= n)
        return;

    as[i * sum_len + sum_len / 2 - 1] += as[i * sum_len - 1];
}