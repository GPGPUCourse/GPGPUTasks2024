__kernel void prefix_sum(__global const int *in, __global int *out, const int offset) {
    unsigned int gid = get_global_id(0);
    out[gid] = in[gid];
    if (gid >= offset)
        out[gid] += in[gid - offset];
}
__kernel void prefix_sum_work_efficient_first(__global int *as, const int offset, const int n) {
    unsigned int gid = get_global_id(0);
    unsigned int right = (gid + 1) * 2 * offset - 1;
    unsigned int left = right - offset;
    if (right < n)
        as[right] = as[left] + as[right];
}
__kernel void prefix_sum_work_efficient_second(__global int *as, const int offset, const int n) {
    unsigned int gid = get_global_id(0);
    unsigned int left = (gid + 1) * 2 * offset - 1;
    unsigned int right = left + offset;
    if (right < n)
        as[right] = as[left] + as[right];
}