__kernel void prefix_sum(__global unsigned int *pr, __global unsigned int *ar, int sz, int n) {
    int index = get_global_id(0);

    if (index >= n) {
        return;
    }

    ar[index] = pr[index]

    if (index >= sz) {
        ar[index] += pr[index - sz]
    }
}
