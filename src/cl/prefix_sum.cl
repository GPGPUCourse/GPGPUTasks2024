__kernel void upsweep(__global unsigned int* data, unsigned int shift, unsigned int n) {
    int gid = get_global_id(0);
    unsigned int index = shift * (gid + 1) - 1;

    if (index < n) {
        data[index] += data[index - shift / 2];
    }
}

__kernel void downsweep(__global unsigned int* data, unsigned int shift, unsigned int n) {
    int gid = get_global_id(0);
    unsigned int index = shift * (gid + 1) - 1;

    if (index < n) {
        unsigned int temp = data[index - shift / 2];
        data[index - shift / 2] = data[index];
        data[index] += temp;
    }
}

__kernel void set_zero(__global unsigned int* data, unsigned int n) {
    if (get_global_id(0) == 0) {
        data[n - 1] = 0;
    }
}
