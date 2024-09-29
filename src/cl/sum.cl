__kernel void sum_atomic(
    __global const unsigned int* arr,
    __global unsigned int* sum,
    unsigned int n
) {
    const size_t index = get_global_id(0);
    if (index >= n) {
        return;
    }
    atomic_add(sum, arr[index]);
}

#define WORKITEM_SIZE 64

__kernel void sum_loop_not_coalesced(
    __global const unsigned int* arr,
    __global unsigned int* sum,
    unsigned int n
) {
    const unsigned int start = get_global_id(0) * WORKITEM_SIZE;
    const unsigned int end = min(start + WORKITEM_SIZE, n);

    unsigned int partial_sum = 0;
    for (unsigned int i = start; i < end; ++i) {
        partial_sum += arr[i];
    }

    atomic_add(sum, partial_sum);
}

__kernel void sum_loop_coalesced(
    __global const unsigned int* arr,
    __global unsigned int* sum,
    unsigned int n
) {
    const unsigned int group_size = get_local_size(0);
    const unsigned int start = get_group_id(0) * group_size * WORKITEM_SIZE + get_local_id(0);
    const unsigned int end = min(start + group_size * WORKITEM_SIZE, n);

    unsigned int partial_sum = 0;
    for (unsigned int i = start; i < end; i += group_size) {
        partial_sum += arr[i];
    }

    atomic_add(sum, partial_sum);
}

#define WORKGROUP_SIZE 128

__kernel void sum_main_thread(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    unsigned int n
) {
    __local unsigned int local_arr[WORKGROUP_SIZE];

    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    if (gid < n) {
        local_arr[lid] = arr[gid];
    } else {
        local_arr[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int partial_sum = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            partial_sum += local_arr[i];
        }
        atomic_add(sum, partial_sum);
    }
}

__kernel void sum_tree(
    __global const unsigned int *arr,
    __global unsigned int *sum,
    unsigned int n
) {
    __local unsigned int local_arr[WORKGROUP_SIZE];

    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    if (gid < n) {
        local_arr[lid] = arr[gid];
    } else {
        local_arr[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = WORKGROUP_SIZE; k > 1; k /= 2) {
        if (lid * 2 < k) {
            int a = local_arr[lid];
            int b = local_arr[lid + k / 2];
            local_arr[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, local_arr[0]);
    }
}
