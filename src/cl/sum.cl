__kernel void sum1(__global unsigned int *a, unsigned int n, __global unsigned int *result)
{
    const unsigned int gid = get_global_id(0);
    if (gid < n) {
        atomic_add(result, a[gid]);
    }
}

#define limit 32

__kernel void sum2(__global const unsigned int *a, unsigned int n, __global unsigned int *result) {
    int gid = get_global_id(0);
    int sum = 0;
    for (int i = 0; i < limit; ++i) {
        unsigned int index = i + limit * gid;
        if (index < n) {
            sum += a[index];
        }
    }
    atomic_add(result, sum);
}

__kernel void sum3(__global const unsigned int *a, unsigned int n, __global unsigned int *result)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);
    int sum = 0;
    for (int i = limit * group_id * local_size + local_id; i < limit * (group_id + 1) * local_size; i += local_size) {
        if (i < n) {
            sum += a[i];
        }
    }

    atomic_add(result, sum);
}

__kernel void sum4(__global unsigned int *a, unsigned int n, __global unsigned int *result)
{
    int gid = get_global_id(0);
    int localId = get_local_id(0);
    int localSize = get_local_size(0);

    __local unsigned int localSum[256]; // Assuming work group size is 256
    localSum[localId] = a[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < localSize; i <<= 1) {
        if (localId % (i * 2) == 0 && localId + i < localSize) {
            localSum[localId] += localSum[localId + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(result, localSum[0]);
    }
}

__kernel void sum5(__global const unsigned int *a, unsigned int n, __global unsigned int *result)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    __local unsigned int buffer[128];
    buffer[local_id] = a[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = limit * 4; i > 1; i /= 2) {
        if (2 * local_id < i) {
            buffer[local_id] += buffer[local_id + i / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(result, buffer[0]);
    }
}
