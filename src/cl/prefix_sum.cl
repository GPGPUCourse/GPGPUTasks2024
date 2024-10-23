__kernel void up_sweep(__global unsigned int *as, __global unsigned int *bs, unsigned int n, int d)
{
    unsigned int idx = get_global_id(0);
    if (idx >= (n >> (d + 1))) {
        return;
    }
    unsigned int k = idx << (d + 1);
    unsigned int k1 = k + (1 << d) - 1;
    unsigned int k2 = k + (1 << (d + 1)) - 1;
    bs[k2] = k2 != get_global_size(0) ? as[k1] + as[k2] : 0;
}

__kernel void down_sweep(__global unsigned int *as, __global unsigned int *bs, unsigned int n, int d)
{
    unsigned int idx = get_global_id(0);
    if (idx >= (n >> (d + 1))) {
        return;
    }
    unsigned int k = idx << (d + 1);
    unsigned int k1 = k + (1 << d) - 1;
    unsigned int k2 = k + (1 << (d + 1)) - 1;
    bs[k1] = as[k2];
    bs[k2] = as[k1] + as[k2];
}
