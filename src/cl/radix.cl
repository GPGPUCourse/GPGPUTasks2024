#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

#define SPAN_SIZE (1 << LOG_SPAN)
#define N_COUNTERS (1 << RADIX_COUNT)
#define MASK (N_COUNTERS - 1)

#define ONES (0xffffffffu)

__kernel void radix_count(__global const unsigned int *as, __global unsigned int* counters, const unsigned int n, const unsigned int shift) {

    const unsigned int gid = get_group_id(0);
    const unsigned int lid = get_local_id(0);

    const unsigned int g_off = gid * SPAN_SIZE;

    __local int loc_counters[N_COUNTERS];
    for (int i = lid; i < N_COUNTERS; i += GROUP_SIZE) {
        loc_counters[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < SPAN_SIZE; i += GROUP_SIZE) {
        const unsigned int x = g_off + i >= n ? ONES : as[g_off + i];
        atomic_add(loc_counters + ((x >> shift) & MASK), 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < N_COUNTERS; i += GROUP_SIZE) {
        counters[N_COUNTERS * gid + i] = loc_counters[i];
    }
}

__kernel void matrix_transpose(__global const unsigned int *as, __global unsigned int *as_t, const unsigned int m, unsigned const int k) {
    unsigned int i = get_local_id(0);
    unsigned int j = get_local_id(1);
    unsigned int wi = TILE_SIZE * get_group_id(0) + i;
    unsigned int wj = TILE_SIZE * get_group_id(1) + j;

    __local unsigned int buf[TILE_SIZE][TILE_SIZE + 1];

    const int stride_j = TILE_SIZE / LINES_PER_GROUP1;
    const int stride_i = TILE_SIZE / LINES_PER_GROUP0;

    for (int dj = 0; dj < LINES_PER_GROUP1; ++dj) {
        for (int di = 0; di < LINES_PER_GROUP0; ++di) {
            const unsigned int off_i = di * stride_i;
            const unsigned int off_j = dj * stride_j;
            buf[i + off_i][j + off_j] = wi + off_i >= k || wj + off_j >= m ? 0 : as[(wj + off_j) * k + wi + off_i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int dj = 0; dj < LINES_PER_GROUP1; ++dj) {
        for (int di = 0; di < LINES_PER_GROUP0; ++di) {
            const unsigned int off_i = di * stride_i;
            const unsigned int off_j = dj * stride_j;
            const unsigned int aj = wi - i + j + off_j;
            const unsigned int ai = wj - j + i + off_i;
            if (ai < m && aj < k) as_t[aj * m + ai] = buf[j + off_j][i + off_i];
        }
    }
}

__kernel void prefix_sum(__global unsigned int *as, const unsigned int off, const unsigned int stride, const unsigned int n) {
    const unsigned int wid = get_global_id(0);
    const unsigned int idx = off + 2 * stride * (wid + 1) - 1;
    if (idx >= n) return;
    as[idx] += as[idx - stride];
}

__kernel void radix_sort(__global const unsigned int *as, __global const unsigned int* pref_counters, __global unsigned int* bs, const unsigned int n, const unsigned int shift) {
    const unsigned int gid = get_group_id(0);
    const unsigned int lid = get_local_id(0);

    const unsigned int g_off = gid * SPAN_SIZE;

    __local int loc_counters[N_COUNTERS];
    __local int loc_offsets[SPAN_SIZE];

    unsigned const int spans = (n + (1 << LOG_SPAN) - 1) >> LOG_SPAN;
    for (int i = lid; i < N_COUNTERS; i += GROUP_SIZE) {
        loc_counters[i] = gid == 0 && i == 0 ? 0 : pref_counters[i * spans + gid - 1];
    }
    if (lid < N_COUNTERS) {
        int off = 0;
        for (int i = 0; i < SPAN_SIZE; ++i) {
            const unsigned int x = g_off + i >= n ? ONES : as[g_off + i];
            const unsigned int key = ((x >> shift) & MASK);
            if (key == lid) {
                loc_offsets[i] = off++;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < SPAN_SIZE; i += GROUP_SIZE) {
        const unsigned int x = g_off + i >= n ? ONES : as[g_off + i];
        const unsigned int key = ((x >> shift) & MASK);
        const unsigned int off = loc_counters[key] + loc_offsets[i];
        if (off <= n) {
            bs[off] = x;
        }
    }
}
