#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

__kernel void prefix_sum_naive(__global const unsigned int *array,
                               __global unsigned int *result,
                               unsigned int chunk_size)
{
    const unsigned int gid = get_global_id(0);

    result[gid] = array[gid];
    if (chunk_size <= gid) {
        result[gid] += array[gid - chunk_size];
    }
}
