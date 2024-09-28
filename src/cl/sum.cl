__kernel void sum_atomic(
    __global unsigned int* result, 
    __global const unsigned int* as, 
    unsigned int size) 
{
    const unsigned int gid = get_global_id(0);
    if (gid < size) {
        atomic_add(result, as[gid]);
    }
}

#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 128

__kernel void sum_for_loop(
    __global unsigned int* result, 
    __global const unsigned int* as, 
    unsigned int size) 
{
    const unsigned int gid = get_global_id(0);

    int local_result = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < size) {
            local_result += as[idx];
        }
    }
    atomic_add(result, local_result);
}

__kernel void sum_for_loop_coalesced(
    __global unsigned int* result, 
    __global const unsigned int* as, 
    unsigned int size) 
{
    const unsigned int lid = get_local_id(0);
    const unsigned int grs = get_local_size(0);
    const unsigned int wid = get_group_id(0);

    int local_result = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < size) {
            local_result += as[idx];
        }
    }
    atomic_add(result, local_result);
}

__kernel void sum_local_mem_single_thread(
    __global unsigned int* result, 
    __global const unsigned int* as, 
    unsigned int size) 
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int grs = get_local_size(0);
    
    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < size ? as[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int local_result = 0;
        for (int i = 0; i < grs; ++i) {
            local_result += buf[i];
        }
        atomic_add(result, local_result);

    }
}

__kernel void sum_local_mem_tree(
    __global unsigned int* result, 
    __global const unsigned int* as, 
    unsigned int size) 
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int grs = get_local_size(0);
    
    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < size ? as[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int local_result = 0;
    for (int jump = 1; jump < WORKGROUP_SIZE; jump *= 2) {
        int idx = 2 * lid * jump;
        if (idx < WORKGROUP_SIZE) {
            buf[idx] += buf[idx + jump];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        atomic_add(result, buf[0]);
    }
}

