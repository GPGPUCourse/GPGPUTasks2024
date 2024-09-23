__kernel void sum_1(__global int *arr, unsigned int n, __global int *sum)
{
    int idx = get_global_id(0);

    if (idx > n) {
        return;
    }

    atomic_add(sum, arr[idx]);
}
