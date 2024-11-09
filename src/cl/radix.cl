#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6
#define WG_SIZE 128
#define SUBMATRIX_SIZE 16


// честно украдено из предыдущей домашки
__kernel void work_efficient_prefix_sum(__global unsigned int *as, const unsigned int s, const unsigned int n, const int desc)
{
    const unsigned int gidx = get_global_id(0);

    const unsigned int left = desc ? ((gidx + 1) * s * 2 - 1) : (gidx * s * 2 + s - 1);
    const unsigned int right = left + s;

    if (right < n)
        as[right] += as[left];
}

__kernel void matrix_transpose_local_good_banks(__global unsigned int* matrix, __global unsigned int* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);
    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    const unsigned int ridx = gidy - lidy + lidx;
    const unsigned int ridy = gidx - lidx + lidy;

    __local unsigned int groupData[SUBMATRIX_SIZE][SUBMATRIX_SIZE + 1];

    const unsigned int stairIndex = (lidx + lidy) % SUBMATRIX_SIZE;
    groupData[lidy][stairIndex] = (gidy < height && gidx < width) ? matrix[gidy * width + gidx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (ridy < width && ridx < height)
        result[ridy * height + ridx] = groupData[lidx][stairIndex];
}


__kernel void make_counters(__global const unsigned int *as, __global unsigned int *counters, const unsigned int pos, const unsigned int nbits)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int groupid = get_group_id(0);

    unsigned int bits = (as[gidx] >> pos) & ((1 << nbits) - 1);
    // coalesced write, transpose later for coalesced read
//    __local unsigned int groupCounters[16];
//    atomic_inc(&groupCounters[bits]);
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//
//    if (lidx < (1 << nbits))
//        counters[groupid * (1 << nbits) + lidx] = groupCounters[lidx];
    atomic_inc(&counters[groupid * (1 << nbits) + bits]);
}


__kernel void set_zeros(__global unsigned int *as)
{
    const unsigned int gidx = get_global_id(0);
    as[gidx] = 0;
}


__kernel void radix_sort(__global const unsigned int *as, __global unsigned int *bs,
                         __global const unsigned int *countersTransposed, const unsigned int pos, const unsigned int nbits)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int groupid = get_group_id(0);
    const unsigned int lidx = get_local_id(0);

    __local unsigned int bitValues[WG_SIZE];
    bitValues[lidx] = (as[gidx] >> pos) & ((1 << nbits) - 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int cidx = bitValues[lidx] * get_num_groups(0) + groupid;
    const unsigned int leftValuesAllWGs = (cidx > 0) ? countersTransposed[cidx - 1] : 0;

    unsigned int sameValueWithinWGCounter = 0;

    for (int i = 0; i < lidx; ++i)
        sameValueWithinWGCounter += bitValues[i] == bitValues[lidx];
    bs[leftValuesAllWGs + sameValueWithinWGCounter] = as[gidx];
}
