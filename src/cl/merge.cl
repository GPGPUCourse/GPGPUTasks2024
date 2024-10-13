#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

size_t binary_search_leftmost(__global const int *as, const size_t n, const int elem) {
    size_t L = 0;
    size_t R = n;
    while (L < R) {
        size_t M = (L + R) / 2;
        if (as[M] >= elem) {
            R = M;
        } else {
            L = M + 1;
        }
    }

    return L;
}

size_t binary_search_rightmost(__global const int *as, const size_t n, const int elem) {
    size_t L = 0;
    size_t R = n;
    while (L < R) {
        size_t M = (L + R) / 2;
        if (as[M] > elem) {
            R = M;
        } else {
            L = M + 1;
        }
    }

    return R;
}

__kernel void merge_global(__global const int *as, __global int *bs, const unsigned int n, const unsigned int block_size)
{
    const size_t gid = get_global_id(0);

    // To prevent out of bounds
    if (gid >= n) {
        return;
    }

    const size_t block_idx = gid / (block_size * 2);
    const size_t elem_idx = gid % (block_size * 2);
    
    const size_t base_offset = block_idx * (block_size * 2);
    
    const int elem = as[gid];

    if (elem_idx < block_size) {
        bs[base_offset + elem_idx + binary_search_leftmost(as + base_offset + block_size, block_size, elem)] = elem;
    }
    else {
        bs[base_offset + elem_idx - block_size + binary_search_rightmost(as + base_offset, block_size, elem)] = elem;
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
