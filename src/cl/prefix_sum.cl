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

__kernel void prefix_sum_up_sweep(__global unsigned int *array,
                                  unsigned int chunk_size,
                                  unsigned int array_size)
{
    const unsigned int gid = get_global_id(0);

    const unsigned int src = gid * chunk_size + chunk_size / 2 - 1;
    const unsigned int dst = (gid + 1) * chunk_size - 1;

    if (array_size <= src || array_size <= dst) {
        return;
    }

    array[dst] += array[src];
}

__kernel void prefix_sum_down_sweep(__global unsigned int *array,
                                    unsigned int chunk_size,
                                    unsigned int array_size)
{
    const unsigned int gid = get_global_id(0);

    const unsigned int src = (gid + 1) * chunk_size - 1;
    const unsigned int dst = (gid + 1) * chunk_size + chunk_size / 2 - 1;

    if (array_size <= src || array_size <= dst) {
        return;
    }

    array[dst] += array[src];
}
