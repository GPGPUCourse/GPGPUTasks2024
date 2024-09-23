__kernel void sum_1(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    int idx = get_global_id(0);

    if (idx >= n) {
        return;
    }

    atomic_add(sum, arr[idx]);
}

#define VALUES_PER_WORKITEM 128
__kernel void sum_2(__global unsigned int *arr, unsigned int n, __global unsigned int *sum)
{
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}
