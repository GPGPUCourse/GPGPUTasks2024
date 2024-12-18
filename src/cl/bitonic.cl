__kernel void bitonic(__global int *a, unsigned int block_size, unsigned int group_size)
{
    unsigned int i = (get_global_id(0) / block_size) * block_size * 2 + get_global_id(0) % block_size;
    unsigned int block_even = (i / group_size) % 2 == 0;

    int temp;
    if (block_even == (a[i] > a[i + block_size])) {
        temp = a[i];
        a[i] = a[i + block_size];
        a[i + block_size] = temp;
    }
}
