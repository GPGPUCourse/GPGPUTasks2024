#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


unsigned int binary_search(__global const int* arr, unsigned int left, unsigned int right, const int value, const bool equal)
{
	while (right > left + 1)
	{
		unsigned int center = (right + left) / 2;

		if ((arr[center] > value) || (equal && (arr[center] == value)))
		{
			right = center;
		}
		else
		{
			left = center;
		}
	}
	return left + 1;
}


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size, unsigned int n)
{
	const unsigned int gidx = get_global_id(0);
	unsigned int blockidx = gidx / block_size;
	unsigned int idx = gidx % block_size;
	unsigned int start = blockidx * block_size * 2;

	unsigned int left = start + block_size - 1;
	unsigned int right = left + block_size + 1;

	int value = as[start + idx];
	unsigned int newidx = binary_search(as, left, right, value, true);
	bs[newidx + idx - block_size] = value;

	left -= block_size;
	right -= block_size;
	value = as[start + idx + block_size];
	newidx = binary_search(as, left, right, value, false);
	bs[idx + newidx] = value;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
