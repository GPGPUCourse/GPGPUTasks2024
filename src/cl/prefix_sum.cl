#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

inline void order(__global int *a, __global int *b, bool ge) {
    int av = *a;
    int bv = *b;

    if ((av >= bv) != ge) {
        *a = bv;
        *b = av;
    }
}

__kernel void prefix_sum1(__global const int *as, __global int *bs, unsigned int stride, unsigned int n) {
    const unsigned int wid = get_global_id(0);
    const unsigned int idx = wid;

    if (idx >= n) return;
    bs[idx] = as[idx] + (idx >= stride ? as[idx - stride] : 0);
}

__kernel void prefix_sum2(__global int *as, unsigned int off, unsigned int stride, unsigned int n) {
    const unsigned int wid = get_global_id(0);
    const unsigned int idx = off + 2 * stride * (wid + 1) - 1;
    if (idx >= n) return;
    as[idx] += as[idx - stride];
}
