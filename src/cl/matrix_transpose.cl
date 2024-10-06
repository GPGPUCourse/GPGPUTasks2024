#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(
    __global float* A,
    __global float* B,
    unsigned int Y /* M */,
    unsigned int X /* K */
) {
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);
    float x = A[gy * X + gx];
    B[gx * Y + gy] = x;
}

#define TILE_SIZE 16
__kernel void matrix_transpose_local_bad_banks(
    __global float* A,
    __global float* B,
    unsigned int Y /* M */,
    unsigned int X /* K */
) {
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    // read from global memory to local and transpose right away
    tile[lx][ly] = A[gy * X + gx];
    barrier(CLK_LOCAL_MEM_FENCE);

    // write to global memory

    // Point at location (gx - lx, gy - ly) -- top-left corner of a tile will go to (gy - ly, gx - lx)
    // If we want to have coalesed write, then we must write from tile from top-left to bottom-right to appropriate locations in B
    // So from top-left corner if we take cell (lx, ly) then it will go to (gy - ly + lx, gx - lx + ly)

    size_t new_gx = gy - ly + lx;
    size_t new_gy = gx - lx + ly;

    B[new_gy * Y + new_gx] = tile[ly][lx];
}

__kernel void matrix_transpose_local_good_banks(
    __global float* A,
    __global float* B,
    unsigned int Y /* M */,
    unsigned int X /* K */
) {
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    // read from global memory to local and transpose right away
    tile[lx][ly] = A[gy * X + gx];
    barrier(CLK_LOCAL_MEM_FENCE);

    // write to global memory
    size_t new_gx = gy - ly + lx;
    size_t new_gy = gx - lx + ly;

    B[new_gy * Y + new_gx] = tile[ly][lx];
}
