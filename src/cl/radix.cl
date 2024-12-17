#define TILE_SIZE 16
#define N_BITS 4
#define WORKGROUP_SIZE 128

__kernel void init_zeros(
    __global int* array,
    unsigned int n
)
{
    int gid = get_global_id(0);
    if (gid < n) {
        array[gid] = 0;
    }
}

__kernel void get_counts(
    __global const int *as,
    __global int *counts,
    unsigned int bit_shift
)
{
    unsigned int gid = get_global_id(0);
    unsigned int grid = get_group_id(0);

    unsigned int value = as[gid];
    unsigned int value_shifted = value >> bit_shift; // remove last non active bits
    unsigned int value_anded = value_shifted & ((1 << N_BITS) - 1); // remove first non active bits

    unsigned int count_ind = (grid << N_BITS) + value_anded;

    atomic_inc(&counts[count_ind]);
}

__kernel void matrix_transpose(
    __global int* a,
    __global int* a_t,
    unsigned int M, unsigned int K
)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    if (y >= M || x >= K) {
        return;
    }

    __local int tile[TILE_SIZE][TILE_SIZE];
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    tile[local_y][local_x] = a[y * K + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_y_T = get_group_id(1) * TILE_SIZE;
    int tile_x_T = get_group_id(0) * TILE_SIZE;

    int in_tile_x_T = local_y;
    int in_tile_y_T = local_x;

    int index = (tile_x_T + in_tile_x_T) * M + (tile_y_T + in_tile_y_T);

    a_t[(tile_x_T + in_tile_x_T) * M + (tile_y_T + in_tile_y_T)] = tile[local_x][local_y];
}

__kernel void pref_sum_we_up(
    __global unsigned int* as,
    __global unsigned int* bs,
    const int d,
    const unsigned int n
) {
    long int gid = get_global_id(0);
    long int idx = (gid + 1) * (1 << d) - 1;
    if (idx < 0 || idx >= n) return;

    unsigned int value = as[idx] + as[idx - (1 << (d - 1))];
    bs[idx] = value;
    as[idx] = value;
}


__kernel void pref_sum_we_down(
    __global unsigned int* as,
    __global unsigned int* bs,
    const int d,
    const unsigned int n
) {
    long int gid = get_global_id(0);
    long int idx = (gid + 1) * (1 << d) - 1 + (1 << (d - 1));
    if (idx < 0 || idx >= n) return;

    unsigned int value = as[idx] + as[idx - (1 << (d - 1))];
    bs[idx] = value;
    as[idx] = value;
}

__kernel void radix_sort(
    __global unsigned int* as,
    __global unsigned int* bs,
    __global int* prefix_counts,
    unsigned int bit_shift,
    unsigned int n
)
{
    unsigned int gid = get_global_id(0);
    unsigned int grid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int ngroups = get_num_groups(0);

    if (gid > n) {
        return;
    }

    __local unsigned int buf[WORKGROUP_SIZE];

    unsigned int value = as[gid];
    value = value >> bit_shift; // remove last non active bits
    value = value & ((1 << N_BITS) - 1); // remove first non active bits
    buf[lid] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int N_eq_before_in_group = 0;
    for (unsigned int i = 0; i < lid; ++i) {
        N_eq_before_in_group += (buf[i] == value);
    }

    unsigned int index_in_counts = value * ngroups + grid;
    int N_leq_in_prev_groups = 0;
    if (index_in_counts > 0 && index_in_counts < n) {
        N_leq_in_prev_groups = prefix_counts[index_in_counts - 1];
    }

    unsigned int index = N_leq_in_prev_groups + N_eq_before_in_group;
    if (index < n) {
        bs[index] = as[gid];
    }
}