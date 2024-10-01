#ifdef CLION_IDE
#endif


#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 64

__kernel void atomic_sum(__global const unsigned int* arr,
                        __global unsigned int* sum,
                         unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    if (gid < n)
        atomic_add(sum, arr[gid]);
}

__kernel void cycle_sum(__global const unsigned int* arr,
                        __global unsigned int* sum,
                        unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        unsigned int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n)
            res += arr[idx];
    }
    atomic_add(sum, res);
}

__kernel void cycle_coalesced_sum(__global const unsigned int* arr,
                                __global unsigned int* sum,
                                unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int grs = get_local_size(0);
    const unsigned int gid = get_group_id(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        unsigned int idx = gid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n)
            res += arr[idx];
    }
    atomic_add(sum, res);
}

__kernel void local_mem_sum(__global const unsigned int* arr,
                                __global unsigned int* sum,
                                unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int grs = get_local_size(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (lid == 0) {    
        int res = 0;
        for (int i = 0; i < grs; i++) {
            res += buf[i];
        }
        atomic_add(sum, res);
    }

}

__kernel void tree_sum(__global const unsigned int* arr, // local group не может быть не степенью 2-ки
                        __global unsigned int* sum,
                        unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int grs = get_local_size(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = grs; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) 
        atomic_add(sum, buf[0]);
}