__kernel void prefix_sum(__global unsigned int *pr, __global unsigned int *ar, int sz, int n) {
    int index = get_global_id(0);

    if (index >= n) {
        return;
    }

    ar[index] = pr[index];

    if (index >= sz) {
        ar[index] += pr[index - sz];
    }
}


__kernel void prefix_sum_up_sweep(__global unsigned int *ar, unsigned int sz, unsigned int ar_sz) {
    int ind = get_global_id(0);

    if (ar_sz > (ind + 1) * sz - 1) {
        ar[(ind + 1) * sz - 1] += ar[ind * sz + sz / 2 - 1];
    }
}

__kernel void prefix_sum_down_sweep(__global unsigned int *ar, unsigned int sz, unsigned int ar_sz) {
    int ind = get_global_id(0);

    if (ar_sz > (ind + 1) * sz + sz / 2 - 1) {
        ar[(ind + 1) * sz + sz / 2 - 1] += ar[(ind + 1) * sz - 1];
    }
}
