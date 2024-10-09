kernel void summation(global uint* numbers,
                      global uint* sum,
                      const uint n,
                      const uint work_group_size,
                      local uint* work_group_buf,
                      const uint num_of_levels) {
    int i = get_global_id(0);
    if (i >= n)
        return;

    int local_i = get_local_id(0);
    uint step = 1;
    uint offset = 0;

    for (uint k = 0; k < num_of_levels; ++k) {
        if (k == 0) {
            work_group_buf[local_i] = numbers[i];
        } else {
            if (local_i % step == 0) {
                work_group_buf[local_i] += work_group_buf[local_i + offset];
            }
        }
        offset *= 2;
        step *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Sum the final results from each work group
    if (local_i == 0) {
        atomic_add(sum, work_group_buf);
    }
}