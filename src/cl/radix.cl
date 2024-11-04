#define check_index(_i, _n, _msg) if ((_i) >= (_n)) printf("%s i: %d n: %d global_id: %d\n", _msg, _i, _n, get_global_id(0))////

__kernel void fill_with_zeros(__global unsigned int *as, unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    //check_index(global_id, n, "fill_with_zeros");////
    if (global_id >= n)
        return;
    //check_index(global_id, n, "fill_with_zeros");////
    as[global_id] = 0;
}

__kernel void work_efficient_sum(__global unsigned int *as, unsigned int n, unsigned int block_size) {
    const unsigned int global_id = get_global_id(0);
    if (global_id * block_size >= n)
        return;
    const unsigned int left_border = global_id * block_size;
    const unsigned int right_border = left_border + block_size - 1;
    const unsigned int middle_border = left_border + block_size / 2 - 1;
    //check_index(right_border, n, "work_efficient_sum right_border");////
    //check_index(middle_border, n, "work_efficient_sum middle_border");////
    as[right_border] += as[middle_border];
}

__kernel void refresh(__global unsigned int *as, unsigned int n, unsigned int block_size) {
    const unsigned int global_id = get_global_id(0);
    const unsigned long long current_source_position = (global_id + 1) * block_size - 1;
    const unsigned long long current_destination_position = current_source_position + block_size / 2;
    if (current_destination_position >= n)
        return;
    //printf("%d\n", block_size);////
    //check_index(current_destination_position, n, "refresh current_destination_position");////
    //check_index(current_source_position, n, "refresh current_source_position");////
    as[current_destination_position] += as[current_source_position];
}

#define operation(_i) ((as[_i] >> bit_shift) & ((1 << nbits) - 1))

__kernel void work_group_counter(__global unsigned int *as, unsigned int n, unsigned int bit_shift, __global unsigned int *counters) {
    unsigned int global_id = get_global_id(0);
    if (global_id >= n)
        return;
    unsigned int work_group_index = get_group_id(0);
    unsigned int sorting_bit = operation(global_id);
    //check_index(work_group_index + work_group_need * sorting_bit, n, "work_group_counter");////
    atomic_inc(&counters[work_group_index + work_group_need * sorting_bit]);
}

__kernel void radix_sort(__global unsigned int *as, unsigned int n, unsigned int bit_shift, __global unsigned int *counters, __global unsigned int *result) {
    unsigned int global_id = get_global_id(0);
    if (global_id >= n)
        return;
    //check_index(global_id, n, "radix_sort sorting_bit");////
    unsigned int work_group_index = get_group_id(0);
    unsigned int sorting_bit = operation(global_id);
    unsigned int index_start = work_group_index * work_group_size;
    unsigned int index_end = global_id;
    unsigned int count_this_eq = 0;
    for (unsigned int i = index_start; i < index_end; ++i) {
        //check_index(i, n, "radix_sort count_this_eq");////
        if (operation(i) == sorting_bit)
            ++count_this_eq;
    }
    unsigned int count_less_and_prev_eq;
    if (work_group_index == 0 && sorting_bit == 0)
        count_less_and_prev_eq = 0;
    else
        count_less_and_prev_eq = counters[work_group_index + work_group_need * sorting_bit - 1];
    //check_index(count_this_eq + count_less_and_prev_eq, n, "radix_sort count_this_eq + count_less_and_prev_eq");////
    result[count_this_eq + count_less_and_prev_eq] = as[global_id];
}
