__kernel void naive_prefix_sum(
    __global const unsigned int *prev,
    __global unsigned int *curr,
    unsigned int k,
    unsigned int n
) {
    int gid = get_global_id(0);

    if (gid >= n) {
        return;
    }

    if (gid < k) {
        curr[gid] = prev[gid];

    } else {
        curr[gid] = prev[gid] + prev[gid - k];
    }
}