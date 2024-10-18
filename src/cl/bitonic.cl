#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

__kernel void bitonic(__global int *a, unsigned int chunk_size_log, unsigned int block_size_log)
{
    unsigned int block_size = 1 << block_size_log;
    unsigned int idx = get_global_id(0);
    unsigned int chunk_id = idx >> chunk_size_log;
    unsigned int block_id = idx >> block_size_log;
    unsigned int block_idx = idx & (block_size - 1);
    bool flip = chunk_id & 1;

    unsigned int i = (block_id << (block_size_log + 1)) + block_idx;
    unsigned int j = i + block_size;

    int ai = a[i];
    int aj = a[j];
    if ((ai > aj) ^ flip) {
        a[j] = ai;
        a[i] = aj;
    }
}
