
#define WORK_GROUP_SIZE 128

__kernel void fill_with_zeros(__global unsigned int *as, unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    as[gid] = 0;
}

__kernel void count(__global unsigned int *as, __global unsigned int *counters, unsigned int n, unsigned int shift, unsigned int bits_count) {
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    unsigned int wgid = get_group_id(0);
    unsigned int bit_idx = (as[gid] >> shift) & ((1 << bits_count) - 1);
    atomic_inc(&counters[wgid + bits_count * bit_idx]);
}

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int i, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    if (gid >= i) {
        bs[gid] = as[gid - i] + as[gid];
    } else {
        bs[gid] = as[gid];
    }
}

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *counters, unsigned int n, unsigned int shift, unsigned int bits_count)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    
    unsigned int wgid = get_group_id(0);
    unsigned int bit_idx = (as[gid] >> shift) & ((1 << bits_count) - 1);
    
    unsigned int start = wgid * WORK_GROUP_SIZE;
    unsigned int end = gid;
    unsigned int local_count = 0;
    
    for (unsigned int i = start; i < end; ++i) {
        unsigned int local_bit_idx = (as[gid] >> shift) & ((1 << bits_count) - 1);
        if (local_bit_idx == bit_idx) {
            ++local_count;
        }
    }
    
    unsigned int prev_count;
    if (wgid == 0 && bit_idx == 0) {
        prev_count = 0;
    } else {
        prev_count = counters[wgid + bits_count * bit_idx - 1];
    }
    
    bs[local_count + prev_count] = as[gid];
}
