#define VALUES_PER_WORK_ITEM 64

__kernel void sum_global_atomic_add(
    __global unsigned int* input,
    __global unsigned int* sum,
    unsigned int n
) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n) {
        return;
    }

    atomic_add(sum, input[gid]);
}

__kernel void sum_cycle(
    __global unsigned int* input,
    __global unsigned int* sum,
    unsigned int n
) {
    const unsigned int gid = get_global_id(0);

    if (gid > (n - VALUES_PER_WORK_ITEM) / VALUES_PER_WORK_ITEM) {
        return;
    }

    unsigned int result = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        const size_t idx = gid * VALUES_PER_WORK_ITEM + i;

        if (idx < n) {
            result += input[idx];
        }
    }

    atomic_add(sum, result);
}
