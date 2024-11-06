#ifndef WORK_SIZE
#error "Constant not defined."
#endif

#ifndef WORK_GROUP_SIZE
#error "Constant not defined."
#endif

#ifndef N_WORK_GROUPS
#error "Constant not defined."
#endif

#ifndef BITS_PER_DIGIT
#error "Constant not defined."
#endif

#ifndef N_DIGITS
#error "Constant not defined."
#endif

#ifndef TILE_SIZE
#error "Constant not defined."
#endif

__kernel void count(__global unsigned int *as, __global unsigned int *cs)
{
    /* TODO */
}

__kernel void transpose(__global unsigned int *as, __global unsigned int *as_t)
{
    unsigned int M = N_WORK_GROUPS;
    unsigned int K = N_DIGITS;

    unsigned int j = get_global_id(0);
    unsigned int i = get_global_id(1);

    __local unsigned int tile[TILE_SIZE + 1][TILE_SIZE];

    unsigned int local_j = get_local_id(0);
    unsigned int local_i = get_local_id(1);

    tile[local_i][local_j] = as[K * i + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[M * j + i] = tile[local_i][local_j];
}

__kernel void up_sweep(__global unsigned int *as, unsigned int n, int d)
{
    unsigned int idx = get_global_id(0);
    if (idx >= (n >> (d + 1))) {
        return;
    }
    unsigned int k = idx << (d + 1);
    unsigned int k1 = k + (1 << d) - 1;
    unsigned int k2 = k + (1 << (d + 1)) - 1;
    as[k2] = k2 != n - 1 ? as[k1] + as[k2] : 0;
}

__kernel void down_sweep(__global unsigned int *as, unsigned int n, int d)
{
    unsigned int idx = get_global_id(0);
    if (idx >= (n >> (d + 1))) {
        return;
    }
    unsigned int k = idx << (d + 1);
    unsigned int k1 = k + (1 << d) - 1;
    unsigned int k2 = k + (1 << (d + 1)) - 1;
    unsigned int tmp = as[k1];
    as[k1] = as[k2];
    as[k2] = tmp + as[k2];
}

__kernel void move()
{
    /* TODO */
}
