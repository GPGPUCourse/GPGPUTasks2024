#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 5

int popcnt32(unsigned int i)
{
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    i = (i + (i >> 4)) & 0x0F0F0F0F;
    i *= 0x01010101;
    return i >> 24;
}


#define MAX_INT ((unsigned int) 0xFFFFFFFF)

#ifdef RADIX_BITS
__kernel void local_radix(
        __global const unsigned int *a,
        const unsigned int n,
        const unsigned int step,
        __global unsigned int *buckets
) {

    const unsigned int gidx = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);
    const unsigned int lidx = get_local_id(0);
    const unsigned int n_buckets = (1 << RADIX_BITS);

    __local unsigned int cnt[4 * (1 << RADIX_BITS)];
    unsigned int shift = 0;
    while(shift + lidx < 4 * n_buckets) {
        cnt[shift + lidx] = 0u;
        shift += group_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int value;
    if (gidx < n){
        value = a[gidx];
    } else{
        value = MAX_INT;
    }
    const unsigned int bucket_idx = (value >> step) & ((1 << RADIX_BITS) - 1);

    const unsigned int pack_idx = lidx / 32u;
    const unsigned int mark = 1 << ((int) lidx - pack_idx * 32);
    atomic_or(&cnt[pack_idx * n_buckets + bucket_idx], mark);
    barrier(CLK_LOCAL_MEM_FENCE);

    shift = 0;
    while (shift + lidx < n_buckets){
        unsigned int bucket_count = 0;
        bucket_count += popcnt32(cnt[0 * n_buckets + shift + lidx]);
        bucket_count += popcnt32(cnt[1 * n_buckets + shift + lidx]);
        bucket_count += popcnt32(cnt[2 * n_buckets + shift + lidx]);
        bucket_count += popcnt32(cnt[3 * n_buckets + shift + lidx]);
        buckets[group_id * n_buckets + shift + lidx] = bucket_count;
        shift += group_size;
    }

}

__kernel void index_radix(
        __global const unsigned int *buckets,
        const unsigned int n,
        const unsigned int step,
        __global const unsigned int *a,
        __global unsigned int *out
) {

    const int gidx = get_global_id(0);
    const int group_id = get_group_id(0);
    const int group_size = get_local_size(0);
    const int lidx = get_local_id(0);
    const int n_buckets = (1 << RADIX_BITS);

    __local unsigned int cnt[4 * (1 << RADIX_BITS)];
    unsigned int shift = 0;
    while(shift + lidx < 4 * n_buckets) {
        cnt[shift + lidx] = 0u;
        shift += group_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int value;
    if(gidx < n){
        value = a[gidx];
    } else {
        value = MAX_INT;
    }
    const unsigned int bucket_idx = (value >> step) & ((1 << RADIX_BITS) - 1);

    const unsigned int pack_idx = lidx >> 5;
    const unsigned int mark = 1 << ((int) lidx - pack_idx * 32);
    atomic_or(&cnt[pack_idx * n_buckets + bucket_idx], mark);
    barrier(CLK_LOCAL_MEM_FENCE);

    int after_count = 0, curr_idx = lidx;
    for (int i = 0; i < 4; ++i) {
        const unsigned int mask = cnt[i * n_buckets + bucket_idx];
        if (curr_idx < 0) {
            after_count += popcnt32(mask);
        }
        else if(curr_idx < 32 - 1) {
            after_count += popcnt32(mask >> (curr_idx + 1));
        }
        curr_idx -= 32;
    }

    const int global_offset = buckets[group_id * n_buckets + bucket_idx];
    const int target_idx = global_offset - after_count - 1;
    if (target_idx < n) {
        out[target_idx] = value;
    }
}
#endif

#if defined(GROUP_SIZE) && defined(USE_PREFIX_SUM)

__kernel void prefix_up(
        __global unsigned int* a,
        const long n,
        const long global_step
){
    const int groups = get_num_groups(0);
    const int group_id = get_group_id(0);
    const int lidx = get_local_id(0);

    __local unsigned int sum[2 * GROUP_SIZE];

    const long start = n - 1 - ((group_id + 1) * 2 * GROUP_SIZE - 1) * global_step;
    const long first_idx = start + lidx * global_step;
    const long second_idx = start + (lidx + GROUP_SIZE) * global_step;

    if(first_idx >= 0){
        sum[lidx] = a[first_idx];
    }else{
        sum[lidx] = 0u;
    }

    if (second_idx >=0){
        sum[lidx + GROUP_SIZE] = a[second_idx];
    } else{
        sum[lidx + GROUP_SIZE] = 0u;
    }

    // LOCAL PREFIX SUM
    int step = 2;
    while (step / 2 < 2 * GROUP_SIZE){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx < 2 * GROUP_SIZE / step){
            int idx = (lidx + 1) * step - 1;
            sum[idx] = sum[idx] + sum[idx - step / 2];
        }
        step *= 2;
    }
    step /= (2 * 2);

    while (step > 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx < 2 * GROUP_SIZE / step - 1) {
            int idx = (lidx + 1) * step + step / 2 - 1;
            sum[idx] = sum[idx] + sum[idx - step / 2];
        }
        step /= 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(first_idx >= 0) a[first_idx] = sum[lidx];
    if(second_idx >= 0) a[second_idx] = sum[lidx + GROUP_SIZE];

}


__kernel void prefix_down(
        __global unsigned int* a,
        const long n,
        const long global_step
){
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    const int lidx = get_local_id(0);

    const long start = n - 1 - ((group_id + 1) * group_size - 1) * global_step;
    const long idx = start + lidx * global_step;

    unsigned int prev_sum = 0;
    if (start >= global_step){
        prev_sum = a[start - global_step];
    }

    if((idx >= 0) && (lidx != group_size - 1)){
        const unsigned int value = a[idx];
        a[idx] = value + prev_sum;
    }

}

#endif

#if defined(TILE_SIZE) && defined(USE_TRANSPOSE)
__kernel void transpose(
        __global const unsigned int *a,
        __global unsigned int *at,
        const unsigned int M,
        const unsigned int K

) {
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    const unsigned int jj = get_local_id(0);
    const unsigned int ii = get_local_id(1);

    __local unsigned int cache[TILE_SIZE][TILE_SIZE];

    unsigned int value = (i < M && j < K) ? a[i * K + j] : 0u;
    cache[jj][(ii + jj) % TILE_SIZE] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int target_j = get_group_id(0) * TILE_SIZE + ii;
    const unsigned int target_i = get_group_id(1) * TILE_SIZE + jj;
    if (target_i < M && target_j < K)
        at[target_j * M + target_i] = cache[ii][(ii + jj) % TILE_SIZE];
}
#endif
