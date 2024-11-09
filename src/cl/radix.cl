#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 5

#define TILE_SIZE 16
#define NUM_BUCKETS 16
#define WG_SIZE 128

bool lttb(unsigned int a, unsigned int b, int i, int j, int shift) {
    unsigned int cmp_a = ((a >> shift) & (NUM_BUCKETS - 1));
    unsigned int cmp_b = ((b >> shift) & (NUM_BUCKETS - 1));
    return cmp_a < cmp_b || (cmp_a == cmp_b && i < j);
}


#define imax(a, b) ((a) < (b) ? (b) : (a))
#define imin(a, b) ((a) < (b) ? (a) : (b))

int calculate_diagonal_local(const __local unsigned int *a, const __local unsigned int *b, int N, int M, int i, int shift) {
    int l = imax(0, i - M), r = imin(N, i) + 1;
    while (r - l > 1) {
        int m = (l + r) / 2;
        unsigned int a_val = a[m - 1], b_val = b[i - m];
        if (lttb(b_val, a_val, 1, 0, shift)) {
            r = m;
        } else {
            l = m;
        }
    }
    return l;
}

void diagonal_merge_local(const __local unsigned int *a, const __local unsigned int *b, __local unsigned int *res, int N, int M, int i, int shift) {
    int l = calculate_diagonal_local(a, b, N, M, i, shift);

    if (l == N) {
        res[i] = b[i - N];
    } else if (i - l == M) {
        res[i] = a[l];
    } else {
        unsigned int a_val = a[l];
        unsigned int b_val = b[i - l];

        res[i] = lttb(a_val, b_val, 0, 1, shift) ? a_val : b_val;
    }
}

__kernel void radix_step_phase1(
    __global unsigned int *a,
    __global int *counts,
    unsigned int n,
    unsigned int shift
) {
    int i = get_global_id(0);
    int gi = get_group_id(0);
    int li = get_local_id(0);

    __local unsigned int a_local[WG_SIZE];
    __local unsigned int b_local[WG_SIZE];
    __local unsigned int *ap = a_local;
    __local unsigned int *bp = b_local;
    ap[li] = a[i];

    barrier(CLK_LOCAL_MEM_FENCE);
    int block_size = 1, two_block_size = 2;
    while (block_size < WG_SIZE) {
        int block_start_offset = li & ~(two_block_size - 1);
        int bi = li - block_start_offset;
        diagonal_merge_local(
            ap + block_start_offset,
            ap + block_start_offset + block_size,
            bp + block_start_offset,
            block_size,
            block_size,
            bi,
            shift
        );
        block_size = two_block_size;
        two_block_size *= 2;
        __local unsigned int *temp = ap;
        ap = bp;
        bp = temp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int val = ap[li];
    a[i] = val;
    unsigned int bucket = (val >> shift) & (NUM_BUCKETS - 1);

    __local int local_counts[NUM_BUCKETS];
    for (int j = li; j < NUM_BUCKETS; j += WG_SIZE) {
        local_counts[j] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&local_counts[bucket]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = li; j < NUM_BUCKETS; j += WG_SIZE) {
        // printf("%d %d %d %d\n", gi, j, local_counts[j], gi * NUM_BUCKETS + j);
        counts[gi * NUM_BUCKETS + j] = local_counts[j];
        // printf("%d %d %d\n", gi * NUM_BUCKETS + j, counts[gi * NUM_BUCKETS + j], local_counts[j]);
    }
}

__kernel void transpose(
    __global unsigned int *a,
    __global unsigned int *a_t,
    unsigned int m,
    unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int gi = get_group_id(0);
    int gj = get_group_id(1);

    __local unsigned int tile[TILE_SIZE][TILE_SIZE + 1];
    tile[lj][li] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    a_t[(gi * TILE_SIZE + lj) * m + gj * TILE_SIZE + li] = tile[li][lj];
}

__kernel void prefix_sum_step(
    __global unsigned int *a,
    int stride_log,
    int global_offset,
    int add_offset,
    int n,
    int skip
) {
    if (get_global_id(0) >= (n >> stride_log)) return;
    int i = (get_global_id(0) << stride_log) + global_offset;
    int j = i + add_offset;
    if ((j & skip) != skip) {
        a[i] += a[j];
    }
}

__kernel void radix_step_phase2(
    __global unsigned int *a,
    __global unsigned int *b,
    __global int *local_counts,
    __global int *counts,
    unsigned int n,
    unsigned int shift
) {
    int i = get_global_id(0);
    int gi = get_group_id(0);
    int li = get_local_id(0);

    unsigned int val = a[i];
    unsigned int bucket = (val >> shift) & (NUM_BUCKETS - 1);

    b[counts[bucket * (n / WG_SIZE) + gi] - local_counts[gi * NUM_BUCKETS + bucket] + li] = val;
}
