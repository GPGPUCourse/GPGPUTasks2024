#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_1(__global unsigned int* data, __global unsigned int* result, int n) {
    int gid = get_global_id(0);

    if (gid < n) {
        atomic_add(result, data[gid]);
    }
}

#define VALUES_PER_WORKITEM 64

__kernel void sum_2(__global unsigned int* data, __global unsigned int* result, int n) {
    int gid = get_global_id(0);
    unsigned int local_sum = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int index = i + VALUES_PER_WORKITEM * gid;
        if (index < n) {
            local_sum += data[index];
        }
    }

    atomic_add(result, local_sum);
}

#define VALUES_PER_WORKITEM 64

__kernel void sum_3(__global unsigned int* data, __global unsigned int* result, int n) {
    int gid = get_global_id(0);
    int loc_id = get_local_id(0);
    int g_id = get_group_id(0);
    int loc_size = get_local_size(0);
    unsigned int local_sum = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int index = VALUES_PER_WORKITEM * g_id * loc_size + i * loc_size + loc_id;
        if (index < n) {
            local_sum += data[index];
        }
    }

    atomic_add(result, local_sum);
}

#define WORKGROUP_SIZE 128
                                  
__kernel void sum_4(__global unsigned int* data, __global unsigned int* result, int n) {
    __local unsigned int local_data[WORKGROUP_SIZE];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    
    unsigned int local_sum = 0;

    local_data[lid] = data[gid];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (lid == 0) {
        unsigned int group_sum = 0;
        for (int i = 0; i < group_size; i++) {
            group_sum += local_data[i];
        }
        atomic_add(result, group_sum);
    }
}

__kernel void sum_5(__global unsigned int* data, __global unsigned int* result, int n) {
    __local unsigned int local_data[WORKGROUP_SIZE];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    local_data[lid] = (gid < n) ? data[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = WORKGROUP_SIZE / 2; stride > 1; stride /= 2) {
        if (2 * lid < stride) {
            local_data[lid] += local_data[lid + stride / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(result, local_data[0]);
    }
}
