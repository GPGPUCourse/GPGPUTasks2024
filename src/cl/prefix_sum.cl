__kernel void up_sweep(__global unsigned int *as, unsigned int n, int d)
{
    unsigned int idx = get_global_id(0);
    if (idx >= (n >> (d + 1))) {
        return;
    }
    unsigned int k = idx << (d + 1);
    unsigned int k1 = k + (1 << d) - 1;
    unsigned int k2 = k + (1 << (d + 1)) - 1;
    as[k2] = k2 != n - 1 ? as[k1] + as[k2] : 0;
}

__kernel void down_sweep(__global unsigned int *as, unsigned int n, int d)
{
    unsigned int idx = get_global_id(0);
    if (idx >= (n >> (d + 1))) {
        return;
    }
    unsigned int k = idx << (d + 1);
    unsigned int k1 = k + (1 << d) - 1;
    unsigned int k2 = k + (1 << (d + 1)) - 1;
	unsigned int tmp = as[k1];
    as[k1] = as[k2];
    as[k2] = tmp + as[k2];
}
