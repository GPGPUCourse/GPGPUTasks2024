
__kernel void tree_sums(__global uint *const as,
                                         uint const log_cur_step,
                                         uint const n)
{
    uint const gid = get_global_id(0);
    
    uint const idx_right = ((gid + 1) << log_cur_step) - 1;
    uint const half_cur_step = (1 << (log_cur_step - 1));
    uint const idx_left = idx_right - half_cur_step;

    if (n <= idx_right)
    {
        return;
    }
    as[idx_right] += as[idx_left];
}

__kernel void prefix_from_tree(__global uint *const as,
                                         uint const log_cur_step,
                                         uint const n)
{
    uint const gid = get_global_id(0);
    
    uint const idx_left = ((gid + 1) << log_cur_step) - 1;
    uint const half_cur_step = (1 << (log_cur_step - 1));
    uint const idx_right = idx_left + half_cur_step;

    if (n <= idx_right)
    {
        return;
    }
    as[idx_right] += as[idx_left];
}
