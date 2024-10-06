#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(
    __global const float* in,
    __global float* out,
    unsigned int M,
    unsigned int K
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= K || j >= M) {
        return;
    }

    out[i * M + j] = in[j * K + i];
}

#define TILE_SIZE 16
__kernel void matrix_transpose_local_bad_banks(
    __global const float* in,
    __global float* out,
    unsigned int M,
    unsigned int K
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= K || j >= M) {
        return;
    }

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float local_buffer[TILE_SIZE][TILE_SIZE];

    local_buffer[local_j][local_i] = in[j * K + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int target_i = i - local_i + local_j;
    const unsigned int target_j = j - local_j + local_i;

    out[target_i * M + target_j] = local_buffer[local_i][local_j];
}
#undef TILE_SIZE

#define TILE_SIZE 16
__kernel void matrix_transpose_local_good_banks(
    __global const float* in,
    __global float* out,
    unsigned int M,
    unsigned int K
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= K || j >= M) {
        return;
    }

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float local_buffer[TILE_SIZE][TILE_SIZE + 1];

    local_buffer[local_j][local_i] = in[j * K + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int target_i = i - local_i + local_j;
    const unsigned int target_j = j - local_j + local_i;

    out[target_i * M + target_j] = local_buffer[local_i][local_j];
}
#undef TILE_SIZE
