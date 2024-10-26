#define WORKGROUP_SIZE (128)

__kernel void prefix_sum(__global unsigned int* as, __global unsigned int* bs, const unsigned int n, const unsigned int step) {
    const unsigned int gid = get_global_id(0);

    if (gid + step >= n) {
        return;
    }

    if (gid < step) {
        bs[gid] = as[gid];
    }

    bs[gid + step] = as[gid] + as[gid + step];
}

__kernel void prefix_sum_work_efficient_up(__global unsigned int* as, const unsigned int step, const unsigned int workSize) {
    const uint gid = get_global_id(0);

    if (gid >= workSize) {
        return;
    }

    as[step * (gid * 2 + 2) - 1] += as[step * (gid * 2 + 1) - 1];
}

__kernel void prefix_sum_work_efficient_down(__global unsigned int* as, const unsigned int n, const unsigned int step, const unsigned int workSize) {
    const uint gid = get_global_id(0);

    if (gid >= workSize) {
        return;
    }

    if (step * (gid + 1) + step / 2 - 1 >= n) {
        return;
    }
    
    as[step * (gid + 1) + step / 2 - 1] += as[step * (gid + 1) - 1];
}