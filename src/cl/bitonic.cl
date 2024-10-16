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

__kernel void bitonic(__global int *as, unsigned int log_block, unsigned int log_stride, unsigned int n) {
    const unsigned int global_id = get_global_id(0);

    const unsigned int stride = 1 << log_stride;
    const unsigned int group_size = stride * 2;
    const unsigned int group = global_id / stride;
    const unsigned int gidx = global_id % stride;
    const unsigned int off = group * group_size + gidx;

    if (off + stride < n) order(as + off, as + off + stride, global_id & (1 << (log_block - 1)));
}
