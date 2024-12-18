__kernel void prefix_sum(__global uint *a, __global uint *b, uint level) {
    uint i = get_global_id(0);
    b[i] = a[i];

    if (i >= level) {
        b[i] += a[i - level];
    }
}

__kernel void prefix_sum_binary(__global uint *a, uint stride, uint n) {
   uint i = (get_global_id(0) + 1) * stride - 1;
   if (i < n)
        a[i] += a[i - stride / 2];
}

__kernel void prefix_sum_efficient(__global uint *a, uint stride, uint n) {
    uint i = (get_global_id(0) + 1) * stride - 1 + stride / 2;
    if (i < n)
        a[i] += a[i - stride / 2];
}