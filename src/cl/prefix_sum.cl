#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void basic_prefix_sum(__global const unsigned int *as, __global unsigned int *bs, const unsigned int step, const unsigned int n)
{
    const unsigned int gidx = get_global_id(0);

    const unsigned int left = gidx - step;
    const unsigned int right = gidx;

    if (gidx >= n)
        return;

    const unsigned int left_value = as[left];
    const unsigned int right_value = as[right];

    if (gidx >= step)
        bs[right] = left_value + right_value;
    else
        bs[gidx] = right_value;
}


__kernel void work_efficient_prefix_sum(__global unsigned int *as, const unsigned int s, const unsigned int n, const int desc)
{
    const unsigned int gidx = get_global_id(0);

    const unsigned int left = (desc) ? ((gidx + 1) * s * 2 - 1) : (gidx * s * 2 + s - 1);
    const unsigned int right = left + s;

    if (right < n)
        as[right] += as[left];
}