#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 5

__kernel void global_prefix_sum(
        const unsigned int n,
        const __global unsigned int* a,
        const unsigned int step,
        __global unsigned int* out
){
    const int gidx = get_global_id(0);
    if (gidx < step){
        out[gidx] = a[gidx];
    } else if (gidx < n){
        out[gidx] = a[gidx] + (gidx - step >= 0 ? a[gidx - step] : 0u);
    }
}

#ifdef GROUP_SIZE

__kernel void prefix_up(
        __global unsigned int* a,
        const int n,
        const int global_step
){
    const int groups = get_num_groups(0);
    const int group_id = get_group_id(0);
    const int lidx = get_local_id(0);


    __local unsigned int sum[2 * GROUP_SIZE];

    const int start = n - 1 - ((group_id + 1) * 2 * GROUP_SIZE - 1) * global_step;
    const int first_idx = start + lidx * global_step;
    const int second_idx = start + (lidx + GROUP_SIZE) * global_step;

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
        const int n,
        const int global_step
){
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    const int lidx = get_local_id(0);

    const int start = n - 1 - ((group_id + 1) * group_size - 1) * global_step;
    const int idx = start + lidx * global_step;

    unsigned int prev_sum = 0;
    if (start >= (int) global_step){
        prev_sum = a[start - global_step];
    }

    if((idx >= 0) && (lidx != group_size - 1)){
        const unsigned int value = a[idx];
        a[idx] = value + prev_sum;
    }

}

#endif