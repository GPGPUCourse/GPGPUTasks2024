#define TILE_SIZE 16
#define NBITS_ELEMENTS 32

#define NBITS 4
#define NDIGITS 1 << NBITS

__kernel void count(
    __global unsigned int* array, 
    __global unsigned int* counters, 
    const unsigned int shift, 
    const unsigned int n
) {
    __local unsigned int local_counters[NDIGITS];

    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int global_size = get_global_size(0);
    unsigned int local_size = get_local_size(0);

    for (int i = lid; i < NDIGITS; i += local_size) {
        local_counters[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = gid; i < n; i += global_size) {
        unsigned int bucket = (array[i] >> shift) & ((NDIGITS) - 1);
        atomic_inc(&local_counters[bucket]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < (NBITS_ELEMENTS / 2)) {
        counters[get_group_id(0) * (NDIGITS) + lid] = local_counters[lid];
    }
}

__kernel void matrix_transpose_local_good_banks(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    unsigned int i = get_global_id(0); // Номер столбца в A
    unsigned int j = get_global_id(1); // Номер строчки в A

    __local float tile[TILE_SIZE * (TILE_SIZE + 1)];

    unsigned int i_local = get_local_id(0);  // Номер столбца в tile
    unsigned int j_local = get_local_id(1);  // Номер строчки в tile

    tile[j_local * TILE_SIZE + i_local] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i_group = get_group_id(0);
    unsigned int j_group = get_group_id(1);

    at[(i_group * TILE_SIZE + j_local) * m + (j_group * TILE_SIZE + i_local)] = tile[i_local * TILE_SIZE + j_local];
}