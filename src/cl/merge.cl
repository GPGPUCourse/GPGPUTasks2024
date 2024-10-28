#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(__global const int *as, unsigned int left, unsigned int right, int value, bool strict_ordering)
{
	while (right > left + 1) {
		unsigned int middle = (right + left) / 2;

		if (as[middle] > value || (strict_ordering && as[middle] == value)) {
			right = middle;
		} else {
			left = middle;
		}
	}

	return left + 1;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    unsigned int block_id     = i / block_size;
    unsigned int block_offset = i % block_size;
    unsigned int offset = 2 * block_size * block_id;

    // Right half:
    unsigned int left_id  = offset + block_size - 1;
    unsigned int right_id = offset + 2 * block_size;
    int value = as[offset + block_offset];

    bs[block_offset - block_size + binary_search(as, left_id, right_id, value, true)] = value;

    // Left half:
    left_id  = offset - 1;
    right_id = offset + block_size;
    value    = as[offset + block_offset + block_size];

    bs[block_offset + binary_search(as, left_id, right_id, value, false)] = value;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}

