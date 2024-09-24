#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_base(
        __global const unsigned int* a,
        const unsigned int n,
        __global unsigned int* out
){

    const unsigned int globalId = get_global_id(0);

    if (globalId >= n)
        return;

    atomic_add(out, a[globalId]);
}

__kernel void sum_cycle(
        __global const unsigned int* a,
        const unsigned int n,
        const unsigned int n_work,
        __global unsigned int* out
){

    const unsigned int globalId = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < n_work; ++i) {
        const unsigned int idx = globalId * n_work + i;
        if(idx < n){
            res += a[idx];
        }
    }

    atomic_add(out, res);
}

__kernel void sum_cycle_coalesce(
        __global const unsigned int* a,
        const unsigned int n,
        const unsigned int n_work,
        __global unsigned int* out
){

    const unsigned int groupSize = get_local_size(0);
    const unsigned int groupId = get_group_id(0);
    const unsigned int localId = get_local_id(0);

    unsigned int res = 0;
    for (int i = 0; i < n_work; ++i) {
        const unsigned int idx = groupSize * groupId * n_work + i * groupSize + localId;
        if(idx < n){
            res += a[idx];
        }
    }

    atomic_add(out, res);
}

#define GROUP_SIZE 128
__kernel void sum_local(
        __global const unsigned int* a,
        const unsigned int n,
        __global unsigned int* out
){

    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int cache[GROUP_SIZE];
    cache[localId] = globalId < n ? a[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int res = 0;
        for (int i = 0; i < GROUP_SIZE; ++i) {
            res += cache[i];
        }
        atomic_add(out, res);
    }
}

#define GROUP_SIZE 128
__kernel void sum_tree(
        __global const unsigned int* a,
        const unsigned int n,
        __global unsigned int* out
){

    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int cache[GROUP_SIZE];
    cache[localId] = globalId < n ? a[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int size = GROUP_SIZE; size > 1; size /= 2) {
        if (2 * localId < size){
            cache[localId] = cache[localId] + cache[localId + size / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (localId == 0) {
        atomic_add(out, cache[0]);
    }
}