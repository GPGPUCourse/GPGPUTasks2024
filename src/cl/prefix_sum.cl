__kernel void prefix_sum(__global const unsigned int* as, __global unsigned int* prefix_sum, unsigned int offset) {
    unsigned int gid = get_global_id(0);

    if (gid < offset) {
        prefix_sum[gid] = as[gid];
        return;
    }

    prefix_sum[gid] = as[gid - offset] + as[gid];
}