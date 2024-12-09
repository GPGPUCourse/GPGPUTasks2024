#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


uint bin_search(__global int const * const as,
                uint const left_idx,
                uint const right_idx,
                int const val,
                bool const is_leq)
{
    uint l = left_idx;
    uint r = right_idx + 1;

    while (l != r - 1) {
        uint const m = (r + l - 1) / 2;
        if (as[m] < val || (is_leq && as[m] == val)) {
            l = m + 1;
        } else {
            r = m + 1;
        }
    }

    return r - 1;
}

__kernel void merge_global(__global int const *const as,
                           __global int *const bs,
                           uint const block_size,
                           uint const n)
{
    uint const gid = get_global_id(0);

    if (n <= gid) {
        return;
    }

    uint const block_idx = gid / block_size;
    bool const is_left_block = (block_idx % 2 == 0);
    
    uint paired_block_left_idx = gid - gid % block_size + (is_left_block ? block_size : -block_size);
    uint paired_block_right_idx = paired_block_left_idx + block_size;
    paired_block_left_idx = n <= paired_block_left_idx ? n : paired_block_left_idx;
    paired_block_right_idx = n <= paired_block_right_idx ? n : paired_block_right_idx;
    
    uint const paired_offset = bin_search(as, paired_block_left_idx, paired_block_right_idx, as[gid], is_left_block) - paired_block_left_idx;
    uint const res_idx = gid + (is_left_block ? 0 : -block_size) + paired_offset;
    
    bs[res_idx] = as[gid];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
