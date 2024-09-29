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

__kernel void sum_cycle_coalesced(
        __global unsigned int* input,
        __global unsigned int* sum,
        unsigned int n
) {
    const unsigned int ggi = get_group_id(0);
    const unsigned int gli = get_local_id(0);
    const unsigned int gls = get_local_size(0);

    if (ggi * gls > (n - VALUES_PER_WORK_ITEM) / VALUES_PER_WORK_ITEM) {
        return;
    }

    unsigned int result = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        const unsigned int idx = ggi * gls * VALUES_PER_WORK_ITEM + gli + gls * i;

        if (idx < n) {
            result += input[idx];
        }
    }

    atomic_add(sum, result);
}

__kernel void sum_local_mem_main_thread(
        __global const unsigned int* input,
        __global unsigned int* sum,
        unsigned int n
) {
    __local unsigned int buff[128];
    const unsigned int ggi = get_global_id(0);
    const unsigned int gli = get_local_id(0);
    const unsigned int gls = get_local_size(0);

    if (ggi < n) {
        buff[gli] = input[ggi];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gli == 0) {
        unsigned int result = 0;

        for (int i = 0; i < gls; i++) {
            result += buff[i];
        }

        atomic_add(sum, result);
    }
}

__kernel void sum_tree(
        __global const unsigned int* input,
        __global unsigned int* sum,
        unsigned int n
) {
    const unsigned int work_group_size = 128;
    __local unsigned int buff[work_group_size];

    const unsigned int ggi = get_global_id(0);
    const unsigned int gli = get_local_id(0);

    if (ggi < n) {
        buff[gli] = input[ggi];
    } else {
        buff[gli] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = work_group_size; i > 1; i /= 2) {
        if (i > 2 * gli) {
            buff[gli] = buff[gli] + buff[gli + i / 2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gli == 0) {
        atomic_add(sum, buff[0]);
    }
}