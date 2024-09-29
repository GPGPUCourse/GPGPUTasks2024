__kernel void sum_global_atomic(__global unsigned int *arr, unsigned int n, __global unsigned int *result)
{
    int index = get_global_id(0);

    if (index < n) {
        atomic_add(result, arr[index]);
    }
}

#define VALUES_PER_WORK_ITEM 64
__kernel void sum_global_atomic_with_loop(__global unsigned int *arr, unsigned int n, __global unsigned int *result)
{
    int global_index = get_global_id(0);

    int local_result = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        int index = global_index * VALUES_PER_WORK_ITEM + i;
        if (index < n) {
            local_result += arr[index];
        }
    }

    atomic_add(result, local_result);
}

__kernel void sum_global_atomic_with_loop_coalesced(__global unsigned int *arr, unsigned int n, __global unsigned int *result)
{
    int local_index = get_local_id(0);
    int group_index = get_group_id(0);
    int group_size  = get_local_size(0);

    int local_result = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        int index = group_index * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_index;
        if (index < n) {
            local_result += arr[index];
        }
    }

    atomic_add(result, local_result);
}

#define WORK_GROUP_SIZE 64
__kernel void sum_local_memory_with_main_thread(__global unsigned int *arr, unsigned int n, __global unsigned int *result)
{
    int global_index = get_global_id(0);
    int local_index  = get_local_id(0);

    __local unsigned int buffer[WORK_GROUP_SIZE];

    buffer[local_index] = arr[global_index];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int group_result = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            group_result += buffer[i];
        }

        atomic_add(result, group_result);
    }
}

__kernel void sum_tree(__global unsigned int *arr, unsigned int n, __global unsigned int *result)
{
    int global_index = get_global_id(0);
    int local_index  = get_local_id(0);

    __local unsigned int buffer[WORK_GROUP_SIZE];

    buffer[local_index] = (global_index < n ? arr[global_index] : 0);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int n_values = WORK_GROUP_SIZE; n_values > 1; n_values /= 2) {
        if (2 * local_index < n_values) {
            unsigned int a = buffer[local_index];
            unsigned int b = buffer[local_index + n_values / 2];
            buffer[local_index] = a + b;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        atomic_add(result, buffer[0]);
    }
}

