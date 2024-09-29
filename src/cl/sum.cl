#define VALUES_PER_WORK_ITEM 32
#define WORKGROUP_SIZE 128

__kernel void atomic_sum(__global const int *array, __global int *sum, unsigned int n) {
    unsigned int index = get_global_id(0);
    
    if (index < n) {
        atomic_add(sum, array[index]);
    }
}

__kernel void loop_sum(__global const int *arr, __global int *sum, unsigned int n) {
    const unsigned int index = get_global_id(0);
    int res = 0;
    for (int i = index * VALUES_PER_WORK_ITEM; i < (index + 1) * VALUES_PER_WORK_ITEM; ++i) {
        if (i < n) {
            res += arr[i];
        }
    }
    atomic_add(sum, res);
}