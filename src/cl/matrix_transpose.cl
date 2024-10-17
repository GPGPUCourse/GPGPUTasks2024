#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void matrix_transpose_naive(__global float *as, __global float *as_t, int M, int K) {
    int globalIdX = get_global_id(0);
    int globalIdY = get_global_id(1);

    if (globalIdX < K && globalIdY < M) {
        as_t[globalIdX * M + globalIdY] = as[globalIdY * K + globalIdX];
    }
}





__kernel void matrix_transpose_local_bad_banks(__global float *as, __global float *as_t, int M, int K) {
    const int LOCAL_SIZE_X = 32;
    const int LOCAL_SIZE_Y = 8;

    __local float tile[LOCAL_SIZE_Y][LOCAL_SIZE_X];

    int globalIdX = get_global_id(0);
    int globalIdY = get_global_id(1);
    int localIdX = get_local_id(0);
    int localIdY = get_local_id(1);
    int groupIdX = get_group_id(0);
    int groupIdY = get_group_id(1);

    // Load data into shared memory
    if (globalIdY * K + globalIdX < M * K) {
        tile[localIdY][localIdX] = as[globalIdY * K + globalIdX];
    }

    // Synchronize threads in the work group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate new global IDs for transposed matrix
    int newGlobalIdX = groupIdX * LOCAL_SIZE_X + localIdX;
    int newGlobalIdY = groupIdY * LOCAL_SIZE_Y + localIdY;

    // Store transposed data from shared memory to global memory
    if (newGlobalIdX < K && newGlobalIdY < M) {
        as_t[newGlobalIdX * M + newGlobalIdY] = tile[localIdY][localIdX];
    }
}



__kernel void matrix_transpose_local_good_banks(__global float *as, __global float *as_t, int M, int K) {
    const int LOCAL_SIZE_X = 32; // Adjust based on your work group size
    const int LOCAL_SIZE_Y = 8;  // Adjust based on your work group size

    __local float tile[LOCAL_SIZE_Y][LOCAL_SIZE_X + 1]; 

    int globalIdX = get_global_id(0);
    int globalIdY = get_global_id(1);
    int localIdX = get_local_id(0);
    int localIdY = get_local_id(1);
    int groupIdX = get_group_id(0);
    int groupIdY = get_group_id(1);

    // Load data into shared memory
    if (globalIdY * K + globalIdX < M * K) {
        tile[localIdY][localIdX] = as[globalIdY * K + globalIdX];
    }

    // Synchronize threads in the work group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate new global IDs for transposed matrix
    int newGlobalIdX = groupIdX * LOCAL_SIZE_X + localIdX;
    int newGlobalIdY = groupIdY * LOCAL_SIZE_Y + localIdY;

    // Store transposed data from shared memory to global memory
    if (newGlobalIdX < K && newGlobalIdY < M) {
        as_t[newGlobalIdX * M + newGlobalIdY] = tile[localIdY][localIdX];
    }
}