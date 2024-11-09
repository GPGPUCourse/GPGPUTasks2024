__kernel void write_zeros(__global unsigned int* as, unsigned int n) {
    int idx = get_global_id(0);
    if (idx < n)
        as[idx] = 0;
}

unsigned int get_radix(unsigned int val, unsigned int nbits, unsigned int bit_shift) {
    return (val >> bit_shift) & ((1 << nbits) - 1);
}

__kernel void counters_by_workgroup(
    __global const unsigned int* as, 
    __global unsigned int* counters, 
    unsigned int n,
    unsigned int nbits,
    unsigned int bit_shift) 
{
    int gid = get_global_id(0);
    int wid = get_group_id(0);
    if (gid < n)
        atomic_inc(&counters[wid * (1 << nbits) + get_radix(as[gid], nbits, bit_shift)]);
}

#define TILE_SIZE 16

__kernel void matrix_transpose(
    __global unsigned int *as, 
    __global unsigned int *as_t, 
    unsigned int M, 
    unsigned int K)
{
    __local unsigned int block[TILE_SIZE][TILE_SIZE + 1];
	
	unsigned int gid0 = get_global_id(0);
	unsigned int gid1 = get_global_id(1);
	unsigned int lid0 = get_local_id(0);
	unsigned int lid1 = get_local_id(1);

	if ((gid0 < M) && (gid1 < K)) {
		unsigned int index_in = gid1 * M + gid0;
		block[lid1][lid0] = as[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	gid0 = get_group_id(1) * TILE_SIZE + lid0;
	gid1 = get_group_id(0) * TILE_SIZE + lid1;
	if ((gid0 < K) && (gid1 < M)) {
		unsigned int index_out = gid1 * K + gid0;
		as_t[index_out] = block[lid0][lid1];
	}
}

__kernel void prefix_sum(
    __global const unsigned int* src, 
    __global unsigned int* dst, 
    unsigned int size, 
    unsigned int start, 
    unsigned int step, 
    unsigned int jump) 
{
    int idx = start + get_global_id(0) * step;
    if (idx < size && idx >= 0) {
        dst[idx] = src[idx] + ((idx >= jump) ? src[idx - jump] : 0);
    }
}

__kernel void radix_sort(
    __global const unsigned int* as, 
    __global unsigned int* bs, 
    __global unsigned int* counters_t,
    __global unsigned int* counters,
    unsigned int n, 
    unsigned int nbits, 
    unsigned int bit_shift,
    unsigned int n_workgroups) 
{
    int gid = get_global_id(0);
    int wid = get_group_id(0);
    int lid = get_local_id(0);
    if (gid < n) {
        int radix = get_radix(as[gid], nbits, bit_shift);

        // number of elements in my workgroup before me and equal to me
        int N1 = 0;
        for (int i = 1; i <= lid; ++i)
            if (get_radix(as[gid - i], nbits, bit_shift) == radix)
                ++N1;
        
        int N2 = ((radix == 0) && (wid == 0)) ? 0 : counters_t[radix * n_workgroups + wid - 1];
        int idx = N1 + N2;
        bs[idx] = as[gid];
    }
}