#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7


__kernel void bitonic(__global int *as, int blocks_size, int sub_blocks_size) {
    if (sub_blocks_size > blocks_size) {
        return;
    }

    int gid = get_global_id(0);

    int block_id = gid / blocks_size;
    int is_desc = (block_id & 1);

    int item_id = (((gid / sub_blocks_size) * 2) * sub_blocks_size) + gid % sub_blocks_size;
    int paired_item_id = item_id + sub_blocks_size;

    if ((as[item_id] < as[paired_item_id]) ^ (!is_desc)) {
        int temp = as[item_id];
        as[item_id] = as[paired_item_id];
        as[paired_item_id] = temp;
    }
}

__kernel void bitonic_on_shifts(__global int *as, int blocks_size_log, int sub_blocks_size_log) {
    if (sub_blocks_size_log > blocks_size_log) {
        return;
    }

    int gid = get_global_id(0);

    int block_id = gid >> blocks_size_log;
    int is_desc = (block_id & 1);

    int item_id = (((gid >> sub_blocks_size_log) << 1) << sub_blocks_size_log) + (gid & ((1 << sub_blocks_size_log) - 1));
    int paired_item_id = item_id + (1 << sub_blocks_size_log);

    if ((as[item_id] < as[paired_item_id]) ^ (!is_desc)) {
        int temp = as[item_id];
        as[item_id] = as[paired_item_id];
        as[paired_item_id] = temp;
    }
}
