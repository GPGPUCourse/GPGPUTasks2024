#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 5


__kernel void
merge_global(__global const int *as, __global int *bs, const unsigned int n, const unsigned int block_size) {
    const int gidx = get_global_id(0);
    const int block_start = gidx / block_size * block_size;
    const int item_idx = gidx - block_start;

    if (gidx < n) {
        const int value = as[gidx];

        int l, r;
        if (item_idx < block_size / 2) {
            l = block_size / 2 - 1, r = block_size;
        } else {
            l = -1, r = block_size / 2;
        }

        while (r - l > 1) {
            int m = (r + l) >> 1;
            if (block_start + m < n) {
                int m_value = as[block_start + m];
                if (m_value < value) l = m;
                else if (item_idx < block_size / 2 && m_value == value) l = m;
                else r = m;
            } else r = m;
        }

        const int target_idx = block_start - block_size / 2 + item_idx + l + 1;
        bs[target_idx] = value;

    }
}

//#define GROUP_SIZE 128
#ifdef GROUP_SIZE
#define MAX_INT (1 << 31) - 1
__kernel void
calculate_indices(__global const int *as, __global int *inds, const unsigned int n, unsigned int block_size) {
    const unsigned int group_idx = get_global_id(0);
    const unsigned int group_start = group_idx * GROUP_SIZE;
    const unsigned int block_start = group_start / block_size * block_size;
    if (GROUP_SIZE < block_size) {

        const int diag_idx = group_start - block_start;
        const int diag_start = max((int) (diag_idx - block_size / 2), (int) 0);
        const int diag_end = min((int) diag_idx, (int) (block_size / 2));
        int l = diag_start - 1, r = diag_end;
        while (r - l > 1) {
            int m = (l + r) >> 1;

            const int left_index = block_start + m;
            const int left_value = left_index < n ? as[left_index] : MAX_INT;

            const int right_index = block_start + block_size / 2 + diag_start + (diag_end - m - 1);
            const int right_value = right_index < n ? as[right_index] : MAX_INT;

            if (left_value <= right_value) l = m;
            else r = m;
        }

        inds[group_idx * 2 + 0] = block_start + l + 1;
        inds[group_idx * 2 + 1] = block_start + block_size / 2 + diag_start + (diag_end - (l + 1));


    } else {
        inds[group_idx * 2 + 0] = group_idx * GROUP_SIZE;
        inds[group_idx * 2 + 1] = group_idx * GROUP_SIZE + GROUP_SIZE / 2;
    }
}

__kernel void merge_local(__global const int *as, __global const int *inds, __global int *bs, const unsigned int n,
                          const unsigned int block_size) {

    const int group_idx = get_group_id(0);
    const int start_x = inds[group_idx * 2 + 0];
    const int start_y = inds[group_idx * 2 + 1];
    int end_x, end_y;
    if (GROUP_SIZE < block_size){
        const int current_block_idx = group_idx * GROUP_SIZE / block_size;
        if ((group_idx + 1) * GROUP_SIZE / block_size == current_block_idx && group_idx < get_num_groups(0) - 1) {
            end_x = inds[(group_idx + 1) * 2 + 0];
            end_y = inds[(group_idx + 1) * 2 + 1];
        }else {
            end_x = current_block_idx * block_size + block_size / 2;
            end_y = current_block_idx * block_size + block_size;
        }
    } else {
        end_x = start_x + GROUP_SIZE / 2;
        end_y = start_y + GROUP_SIZE / 2;
    }
    const int size_x = end_x - start_x;
    const int size_y = end_y - start_y;

    const int lidx = get_local_id(0);

    __local int cache[GROUP_SIZE];
    int value;
    if (lidx < size_x) {
        value = start_x + lidx < n ? as[start_x + lidx] : MAX_INT;
    } else {
        value = start_y + (lidx - size_x) < n ? as[start_y + (lidx - size_x)] : MAX_INT;
    }
    cache[lidx] = value;
    barrier(CLK_LOCAL_MEM_FENCE);


    const int local_block_size = GROUP_SIZE < block_size ? GROUP_SIZE : block_size;
    const int local_size_x = GROUP_SIZE < block_size ? size_x : local_block_size / 2;
    const int block_start = lidx / local_block_size * local_block_size;
    const int item_idx = lidx - block_start;

    int l, r;
    if (item_idx < local_size_x) {
        l = local_size_x - 1, r = local_block_size;
    } else {
        l = -1, r = local_size_x;
    }

    while (r - l > 1) {
        int m = (r + l) >> 1;
        int m_value = cache[block_start + m];
        if (m_value < value) l = m;
        else if (item_idx < local_size_x && m_value == value) l = m;
        else r = m;
    }

    const int target_idx = block_start - local_size_x + item_idx + l + 1;

    barrier(CLK_LOCAL_MEM_FENCE);
    cache[target_idx] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    int global_target_idx;
    if (GROUP_SIZE < block_size) {
        global_target_idx = start_x + start_y - block_size / 2 - (get_global_id(0) / block_size * block_size);
    } else {
        global_target_idx = start_x;
    }

    if (global_target_idx + lidx < n) {
        bs[global_target_idx + lidx] = cache[lidx];
    }

}
#endif