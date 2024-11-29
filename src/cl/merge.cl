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

__kernel void merge_global(__global float *as, __global float *out, unsigned int block_size, unsigned int n)
{
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    unsigned int block_id = i / block_size;

    int left = 0;
    if (block_id & 1) {
        left = (block_id + 1) * block_size;
    } else {
        left = (block_id - 1) * block_size;
    }

    unsigned int block_offset = i % block_size;
    unsigned int index = binary_search(as, left, left + block_size, as[i], block_id & 1);

    out[block_offset + (block_id - (block_id & 1)) * block_size + index - left] = as[i];
}

