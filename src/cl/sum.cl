#define WORK_PER_ITEM 32
#define WORKGROUP_SIZE 128

__kernel void atomic_sum(__global const int *array, __global unsigned int *sum, unsigned int n) {
    unsigned int index = get_global_id(0);
    
    if (index < n) {
        atomic_add(sum, array[index]);
    }
}

__kernel void loop_sum(__global const int *array, __global unsigned int *sum, unsigned int n) {
    const unsigned int index = get_global_id(0);
    int res = 0;
    for (int i = index * WORK_PER_ITEM; i < (index + 1) * WORK_PER_ITEM; ++i) {
        if (i < n) {
            res += array[i];
        }
    }
    atomic_add(sum, res);
}

__kernel void loop_coalesced_sum(__global const int *array, __global unsigned int *sum, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (unsigned int i = 0; i < WORK_PER_ITEM; i++) {
        unsigned int idx = wid * grs * WORK_PER_ITEM + i * grs + lid;
        if (idx < n) {
            res += array[idx];
        }
    }
    atomic_add(sum, res);
}

__kernel void sum_local_mem(__global const unsigned int* array, __global unsigned int* sum, unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int local_buffer[WORKGROUP_SIZE];

    local_buffer[lid] = gid < n ? array[gid] : 0;

    // Этот барьер синхронизирует все потоки внутри рабочей группы, пока все данные будут загружены в локальную память 
    barrier(CLK_LOCAL_MEM_FENCE);

    // Суммирование элементов из локальной памяти выполняется только первым потоком рабочей группы
    if (lid != 0) {
        return;
    }
    unsigned int group_res = 0;
    for (unsigned int i = 0; i < WORKGROUP_SIZE; i++) {
        group_res += local_buffer[i];
    }
    atomic_add(sum, group_res);
}