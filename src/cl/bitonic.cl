#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 5

#define MAX_INT 2147483647

__kernel void bitonic(__global int* a,
                      const unsigned int n,
                      const unsigned int block_size,
                      const unsigned int local_block_size
                      )
{
    const int worker_idx = get_global_id(0);
    const int local_block_start = worker_idx / local_block_size * local_block_size ;
    const int gidx = local_block_start + worker_idx;
    const int pair_idx = local_block_start + worker_idx + local_block_size;

    const int curr_value = gidx < n ? a[gidx] : MAX_INT;
    const int pair_value = pair_idx < n ? a[pair_idx] : MAX_INT;
    const bool dir = (gidx & block_size) == 0;
    if (curr_value > pair_value == dir) {
        a[gidx] = pair_value;
        a[pair_idx] = curr_value;
    }
}
