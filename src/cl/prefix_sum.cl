__kernel void  no_work_efficient_prefix_sum(
        __global const unsigned int *as,
        __global unsigned int *bs,
        unsigned int k,
        unsigned int n
) {
    int gi = get_global_id(0);
    if (gi >= n)
        return;
    if (gi < k) {
        bs[gi] = as[gi];
    } else {
        bs[gi] = as[gi] + as[gi - k];
    }
}