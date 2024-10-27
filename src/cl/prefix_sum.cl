
__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int i, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    if (gid >= i) {
        bs[gid] = as[gid - i] + as[gid];
    } else {
        bs[gid] = as[gid];
    }
}
