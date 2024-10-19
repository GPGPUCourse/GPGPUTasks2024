#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global int* array, unsigned int block_size, unsigned int arr_len)
{
    int gid = get_global_id(0);
    int group_id = gid / block_size;
    int arr_block_id = gid / arr_len;
    int idx = arr_block_id * 2 * arr_len + gid % arr_len;
    int cf = group_id % 2 == 0 ? 1 : -1;

    if (cf * array[idx] > cf * array[idx + arr_len]) {
        int tmp = array[idx];
        array[idx] = array[idx + arr_len];
        array[idx + arr_len] = tmp;
    }
}
