__kernel void prefix_sum(__global unsigned int* as_gpu, unsigned int offset, unsigned int n, int down) {
    unsigned int gid = get_global_id(0);

    unsigned int index = gid * offset + offset - 1;
    if (down) {
        index -= offset / 2;
    }

    if (index < n) {
        as_gpu[index] += as_gpu[index - offset / 2];
    }
}
