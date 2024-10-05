#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float* left, __global float* right, __global float* result,
                                          const unsigned int M, const unsigned int K, const unsigned int N)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);

    if (gidx >= M || gidy >= N)
        return;

    float result_ = 0;

    for (int i = 0; i < K; i++)
        result_ += left[gidx * M + i] * right[i * N + gidy];

    result[gidx * N + gidy] = result_;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float* left, __global float* right, __global float* result,
                                          const unsigned int M, const unsigned int K, const unsigned int N)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);

    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    const unsigned int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    __local float leftTile[TILE_SIZE][TILE_SIZE];
    __local float rightTile[TILE_SIZE][TILE_SIZE];

    float sum = 0;

    for (int t = 0; t < numTiles; t++)
    {
        leftTile[lidy][lidx] = left[gidy * K + t * TILE_SIZE + lidx];
        rightTile[lidy][lidx] = right[(t * TILE_SIZE + lidy) * N + gidx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++)
        {
            sum += leftTile[lidy][i] * rightTile[i][lidx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[gidy * N + gidx] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float* left, __global float* right, __global float* result,
                                              const unsigned int M, const unsigned int K, const unsigned int N)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);

    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    const unsigned int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    __local float leftTile[TILE_SIZE][TILE_SIZE];
    __local float rightTile[TILE_SIZE][TILE_SIZE];

    float wpt_accumulator[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; w++)
        wpt_accumulator[w] = 0;

    for (int t = 0; t < numTiles; t++)
    {
        for (int w = 0; w < WORK_PER_THREAD; w++)
        {
            leftTile[lidy * WORK_PER_THREAD + w][lidx] = left[(gidy * WORK_PER_THREAD + w) * K + t * TILE_SIZE + lidx];
            rightTile[lidy * WORK_PER_THREAD + w][lidx] = right[(t * TILE_SIZE + lidy * WORK_PER_THREAD + w) * N + gidx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++)
        {
            for (int w = 0; w < WORK_PER_THREAD; w++)
            {
                wpt_accumulator[w] += leftTile[lidy * WORK_PER_THREAD + w][i] * rightTile[i][lidx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WORK_PER_THREAD; w++)
        result[(gidy * WORK_PER_THREAD + w) * N + gidx] = wpt_accumulator[w];
}
#endif
