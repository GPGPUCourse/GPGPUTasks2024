#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6
#define SUBMATRIX_SIZE 16

__kernel void matrix_transpose_naive(__global float* matrix, __global float* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);

    result[gidx * height + gidy] = matrix[gidy * width + gidx];
}

__kernel void matrix_transpose_local_bad_banks(__global float* matrix, __global float* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);
    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

//    if (gidx + gidy == 0)
//    printf(" (%d, %d) ", lidx, lidy);

    // https://nanoreview.net/en/gpu/geforce-rtx-3090
    // 128KB на compute unit, которых 82 штуки, всего 10496 потока. Зная, что в варп nVidia кладет 32 потока получаем 4 варпа на compute unit
    // Значит, один варп может претендовать на 32KB памяти. Посчитал это все, но выделю 4KB как было на лекции, мало ли...
    __local float groupData[SUBMATRIX_SIZE][SUBMATRIX_SIZE];

    groupData[lidy][lidx] = matrix[gidy * width + gidx];

    barrier(CLK_LOCAL_MEM_FENCE);

    float temp = groupData[lidy][lidx];
    groupData[lidy][lidx] = groupData[lidx][lidy];

    barrier(CLK_LOCAL_MEM_FENCE);

    groupData[lidx][lidy] = temp;
    result[gidx * width + gidy] = groupData[lidx][lidy];
}

__kernel void matrix_transpose_local_good_banks(__global float* matrix, __global float* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);
    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    __local float groupData[SUBMATRIX_SIZE][SUBMATRIX_SIZE];

    const unsigned int stairIndex = (lidx + lidy) % SUBMATRIX_SIZE;
    groupData[lidy][stairIndex] = matrix[gidy * width + gidx];

    barrier(CLK_LOCAL_MEM_FENCE);

    float temp = groupData[lidy][stairIndex];
    groupData[lidy][stairIndex] = groupData[stairIndex][lidy];

    barrier(CLK_LOCAL_MEM_FENCE);

    groupData[stairIndex][lidy] = temp;
    result[gidx * width + gidy] = groupData[stairIndex][lidy];
}
