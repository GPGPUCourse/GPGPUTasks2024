__kernel void naive_prefix_sum(
    __global const unsigned int *prev,
    __global unsigned int *curr,
    const unsigned int k,
    const unsigned int n
) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n) {
        return;
    }

    if (gid < k) {
        curr[gid] = prev[gid];

    } else {
        curr[gid] = prev[gid] + prev[gid - k];
    }
}

__kernel void prefix_sum_up_sweep(
    __global unsigned int *array, 
    const unsigned int chunk_size, 
    const unsigned int n
) {
    const unsigned int gid = get_global_id(0);

    const unsigned int src = gid * chunk_size + chunk_size / 2 - 1;
    const unsigned int dst = (gid + 1) * chunk_size - 1;

    if (n > dst) {
        array[dst] += array[src];
    }
}

__kernel void prefix_sum_down_sweep(
    __global unsigned int *array, 
    const unsigned int chunk_size, 
    const unsigned int n
) {
    int gid = get_global_id(0);

    const unsigned int src = (gid + 1) * chunk_size - 1;
    const unsigned int dst = (gid + 1) * chunk_size + chunk_size / 2 - 1;

    if (n > dst) {
        array[dst] += array[src];
    }
}