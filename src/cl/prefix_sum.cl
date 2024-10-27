__kernel void reduce(__global const unsigned int* a, __global unsigned int* b) {
    int i = get_global_id(0);

    b[i] = a[2 * i] + a[2 * i + 1];
}

__kernel void down_sweep(__global const unsigned int* a, __global unsigned int* b) {
    int i = get_global_id(0);

    if (i == 0) {
        return;
    }

    if (i % 2) {
        b[i] = a[i / 2];
    } else {
        b[i] += a[i / 2 - 1];
    }
}