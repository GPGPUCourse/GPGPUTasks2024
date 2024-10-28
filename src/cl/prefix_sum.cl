#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

#define u32 unsigned int

__kernel void prefix_sum_step2(
    __global u32 *a,
    __global u32 *b,
    int global_offset,
    int add_offset,
    int n
) {
    int i = get_global_id(0) + global_offset;
    if (i >= n) return;
    int j = i + add_offset;
    b[i] = a[i] + a[j];
}

__kernel void prefix_sum_step(
    __global u32 *a,
    int stride_log,
    int global_offset,
    int add_offset,
    int n
) {
    int i = (get_global_id(0) << stride_log) + global_offset;
    if (i >= n) return;
    int j = i + add_offset;
    a[i] += a[j];
}

/*
u32 reverse(u32 x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16);
    return x;
}

__kernel void permute(
    __global u32 *a,
    int required_shift
) {
    int i = get_global_id(0);
    int j = reverse(i) >> required_shift;
    if (i < j) {
        u32 ai = a[i];
        u32 aj = a[j];
        a[i] = aj;
        a[j] = ai;
    }
}
*/