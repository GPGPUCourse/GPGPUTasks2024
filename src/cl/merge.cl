#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

unsigned int binary_search(__global const int *arr, unsigned int left, unsigned int right, int value) {
    while (left < right) {
        unsigned int mid = left + (right - left) / 2;
        if (arr[mid] <= value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size) {
    int gid = get_global_id(0);
    unsigned int left = gid * block_size;
    unsigned int mid = min((unsigned int)(left + block_size / 2), (unsigned int)get_global_size(0));
    unsigned int right = min((unsigned int)(left + block_size), (unsigned int)get_global_size(0));

    unsigned int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        unsigned int insert_pos = binary_search(as, j, right, as[i]);
        while (j < insert_pos) {
            bs[k++] = as[j++];
        }
        bs[k++] = as[i++];
    }

    while (i < mid) {
        bs[k++] = as[i++];
    }

    while (j < right) {
        bs[k++] = as[j++];
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
