#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum(__global const unsigned int* array, __global unsigned int* prefix_sum, unsigned int offset)
{
    int gid = get_global_id(0);

    if (gid < offset) {
        prefix_sum[gid] = array[gid];
        return;
    }

    prefix_sum[gid] = array[gid - offset] + array[gid];
}

__kernel void prefix_sum_work_eff_down(__global unsigned int* prefix_sum, unsigned int n, unsigned int power)
{
    int gid = get_global_id(0);

    int idx = 2 * (gid + 1) * power - 1;
    if (idx < 0 || idx >= n) {
        return;
    }

    prefix_sum[idx] += prefix_sum[idx - power];
}

__kernel void prefix_sum_work_eff_up(__global unsigned int* prefix_sum, unsigned int n, unsigned int power)
{
    int gid = get_global_id(0);
    int idx = power + 2 * (gid + 1) * power - 1;

    if (idx < 0 || idx >= n) {
        return;
    }

    prefix_sum[idx] += prefix_sum[idx - power];
}
