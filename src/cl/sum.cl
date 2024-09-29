#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TASK_SIZE 96
#define WORKGROUP_SIZE 128

__kernel void global_atomic_sum(__global unsigned int* nums, __global unsigned int* result, const unsigned int n)
{
    const unsigned int idx = get_global_id(0);

    if (idx >= n)
        return;

    atomic_add(result, nums[idx]);
}

__kernel void global_looped_sum(__global unsigned int* nums, __global unsigned int* result, const unsigned int n)
{
    const unsigned int idx = get_global_id(0);

    unsigned int internal_sum = 0;

    for (int i = 0; i < TASK_SIZE; i++)
    {
        if (i * get_global_size(0) + idx >= n) break;
        internal_sum += nums[i * get_global_size(0) + idx];
    }
    atomic_add(result, internal_sum);
}

__kernel void global_looped_coalesced_sum(__global unsigned int* nums, __global unsigned int* result, const unsigned int n)
{
    const unsigned int idx = get_global_id(0);

    unsigned int internal_sum = 0;

    for (int i = idx * TASK_SIZE; i < (idx + 1) * TASK_SIZE; i++)
    {
        if (i >= n) break;
        internal_sum += nums[i];
    }
    atomic_add(result, internal_sum);
}

__kernel void local_mem_sum(__global unsigned int* nums, __global unsigned int* result, const unsigned int n)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int lidx = get_local_id(0);

    __local unsigned int groupData[WORKGROUP_SIZE];

    groupData[lidx] = gidx < n ? nums[gidx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx == 0)
    {
        unsigned int groupResult = 0;
        for (int i = 0; i < WORKGROUP_SIZE; ++i)
            groupResult += groupData[i];
        atomic_add(result, groupResult);
    }
}

__kernel void tree_local_mem_sum(__global unsigned int* nums, __global unsigned int* result, const unsigned int n)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int lidx = get_local_id(0);

    __local unsigned int groupData[WORKGROUP_SIZE];

    groupData[lidx] = gidx < n ? nums[gidx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < (int)ceil(log2((float)WORKGROUP_SIZE)); ++i)
    {
        unsigned int step = (int)pow(2.0f, i);
        if ((lidx % step == 0) && (lidx / step % 2 == 0))
        {
            groupData[lidx] = groupData[lidx] + groupData[lidx + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidx == 0)
    {
        atomic_add(result, groupData[0]);
    }
}
