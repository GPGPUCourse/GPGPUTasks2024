#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

#define WGS 64
#define BTS 4

__kernel void radix(__global const uint *a, __global uint *b, __global uint *counters, __global uint *pref, uint shift) {
    uint i = get_global_id(0);

    __local uint local_counters[1 << BTS];
    for (int j = 0; j < (1 << BTS); j++) {
        local_counters[j] = 0;
    }

    for (int j = 0; j < WGS; j++) {
        uint index = i * WGS + j;
        uint value = (a[index] >> shift) & 0xF;
        uint p = value * get_num_groups(0) + get_group_id(0);
        uint pos = pref[p] - counters[p] + local_counters[value];
        local_counters[value]++;
        b[pos] = a[index];
    }
}

__kernel void counters(__global const uint *a, __global uint *counters, uint shift) {
    __local uint local_counters[WGS];
    uint i = get_local_id(0);
    local_counters[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint index = get_global_id(0);

    uint value = (a[index] >> shift) & 0xF;
    atomic_inc(&local_counters[value]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < (1 << BTS)) {
        counters[i * get_num_groups(0) + get_group_id(0)] = local_counters[i];
    }
}

__kernel void copy_arrays(__global const uint *a, __global uint *b, uint n) {
    uint i = get_global_id(0);
    if (i < n) {
        b[i] = a[i];
    }
}

__kernel void prefix_bin(__global uint *a, uint stride, uint n) {
   uint i = (get_global_id(0) + 1) * stride - 1;
   if (i < n)
        a[i] += a[i - stride / 2];
}

__kernel void prefix(__global uint *pref, uint stride, uint n) {
    uint i = (get_global_id(0) + 1) * stride - 1 + stride / 2;
    if (i < n)
        pref[i] += pref[i - stride / 2];
}