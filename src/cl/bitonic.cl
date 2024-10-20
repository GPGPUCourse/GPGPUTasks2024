#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif
#line 4

__kernel void bitonic(__global int *m, unsigned int chunk_log, unsigned int block_log)
{
    unsigned int global_i = get_global_id(0);

    unsigned int chunk = global_i >> chunk_log;
    unsigned int block = global_i >> block_log;

    unsigned int local_i = global_i % (1 << block_log);

    unsigned int i = (block << (block_log + 1)) + local_i;
    unsigned int j = i + (1 << block_log);

    bool activate = chunk % 2;
    bool lower = m[i] > m[j];

    if ((activate && !lower) || (!activate && lower)) {
        unsigned int tmp = m[j];
        m[j] = m[i]; m[i] = tmp;
    }
}