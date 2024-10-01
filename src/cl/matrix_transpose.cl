#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(
    __global const float* as,
    __global float* as_t,
    int M,
    int K)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    if (gid0 < M && gid1 < K) {
        as_t[gid1 * M + gid0] = as[gid0 * K + gid1];
    }
}

#define TILE_SIZE 16

__kernel void matrix_transpose_local_bad_banks(
    __global const float* as,
    __global float* as_t,
    int M,
    int K)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);

    __local float buf[TILE_SIZE][TILE_SIZE];

    buf[lid1][lid0] = as[gid0 * K + gid1];

    barrier(CLK_LOCAL_MEM_FENCE);

    gid0 += lid1 - lid0;
    gid1 += lid0 - lid1;

    as_t[gid1 * M + gid0] = buf[lid0][lid1];
}

__kernel void matrix_transpose_local_good_banks(
    __global const float* as,
    __global float* as_t,
    int M,
    int K)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);

    __local float buf[TILE_SIZE][TILE_SIZE];

    buf[lid1][(lid0 + lid1) % TILE_SIZE] = as[gid0 * K + gid1];

    barrier(CLK_LOCAL_MEM_FENCE);

    gid0 += lid1 - lid0;
    gid1 += lid0 - lid1;

    as_t[gid1 * M + gid0] = buf[lid0][(lid1 + lid0) % TILE_SIZE];
}
