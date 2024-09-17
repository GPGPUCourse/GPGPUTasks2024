#ifdef __CLION_IDE__
#include "clion_defines.cl"

#endif
#line 8
__kernel void aplusb(
        __global float *a,
        __global float *b,
        __global float *c,
        unsigned int n
) {

    int idx = get_global_id(0);
    if (idx >= n)
        return;

    *(c + idx) = *(a + idx) + *(b + idx);
}
