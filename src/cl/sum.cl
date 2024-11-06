__kernel void sum_1(__global unsigned int* data, __global unsigned int* result, int n) {
    int gid = get_global_id(0);

    if (gid < n) {
        atomic_add(result, data[gid]);
    }
}

__kernel void sum_2(__global unsigned int* data, __global unsigned int* result, int n) {
    int gid = get_global_id(0);
    unsigned int local_sum = 0;

    for (int i = gid; i < n; i += get_global_size(0)) {
        local_sum += data[i];
    }

    atomic_add(result, local_sum);
}

__kernel void sum_3(__global unsigned int* data, __global unsigned int* result, int n) {
    int gid = get_global_id(0);
    unsigned int local_sum = 0;

    for (int i = gid * 32; i < n; i += get_global_size(0) * 32) {
        for (int j = 0; j < 32; ++j) {
            if (i + j < n)
                local_sum += data[i + j];
        }
    }

    atomic_add(result, local_sum);
}

__kernel void sum_4(__global unsigned int* data, __global unsigned int* result, int n) {
    __local unsigned int local_data[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    
    unsigned int local_sum = 0;
    
    for (int i = gid; i < n; i += get_global_size(0)) {
        local_sum += data[i];
    }
    
    local_data[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_sum = 0;
        for (int i = 0; i < group_size; ++i) {
            group_sum += local_data[i];
        }
        atomic_add(result, group_sum);
    }
}

__kernel void sum_5(__global unsigned int* data, __global unsigned int* result, int n) {
    __local unsigned int local_data[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    local_data[lid] = (gid < n) ? data[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            local_data[lid] += local_data[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(result, local_data[0]);
    }
}
