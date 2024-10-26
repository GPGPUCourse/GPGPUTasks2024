__kernel void prefix_sum(__global unsigned int* src, __global unsigned int* dst, int size, int start, int step, int jump) {
    int idx = start + get_global_id(0) * step;
    if (idx < size && idx >= 0) {
        dst[idx] = src[idx] + ((idx >= jump) ? src[idx - jump] : 0);
    }
}