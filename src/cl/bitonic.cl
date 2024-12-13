__kernel void bitonic(__global int *const as,
                      uint const top_block_size,
                      uint const cur_block_size) 
{
    uint const gid = get_global_id(0);
    
    uint const top_block_id = gid / (top_block_size / 2);
    uint const cur_step = cur_block_size / 2;
    uint const cur_block_id = gid / cur_step;
    
    bool const is_order_increase = top_block_id % 2 == 0;
    uint const idx1 = gid + (cur_block_id * cur_step);
    uint const idx2 = idx1 + cur_step;
    
    int const left_value = as[idx1];
    int const right_value = as[idx2];
    bool const is_lesser = left_value < right_value;
    
    if ((is_order_increase && !is_lesser) || (!is_order_increase && is_lesser)) 
    {
        as[idx1] = right_value;
        as[idx2] = left_value;
    }
}

