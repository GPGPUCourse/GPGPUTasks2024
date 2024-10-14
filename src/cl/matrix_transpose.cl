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

__kernel void matrix_transpose_local_bad_banks_non_coalesced(__global float* matrix, __global float* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);
    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    // https://nanoreview.net/en/gpu/geforce-rtx-3090
    // 128KB на compute unit, которых 82 штуки, всего 10496 потока. Зная, что в варп nVidia кладет 32 потока получаем 4 варпа на compute unit
    // Значит, один варп может претендовать на 32KB памяти. Посчитал это все, но выделю 4KB как было на лекции, мало ли...
    __local float groupData[SUBMATRIX_SIZE][SUBMATRIX_SIZE];

    groupData[lidy][lidx] = matrix[gidy * width + gidx];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Заодно тут убрал транспонирование в локальной памяти, зачем я его добавил вообще?

    result[gidx * height + gidy] = groupData[lidy][lidx];
}

__kernel void matrix_transpose_local_bad_banks(__global float* matrix, __global float* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);
    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    // https://nanoreview.net/en/gpu/geforce-rtx-3090
    // 128KB на compute unit, которых 82 штуки, всего 10496 потока. Зная, что в варп nVidia кладет 32 потока получаем 4 варпа на compute unit
    // Значит, один варп может претендовать на 32KB памяти. Посчитал это все, но выделю 4KB как было на лекции, мало ли...
    __local float groupData[SUBMATRIX_SIZE][SUBMATRIX_SIZE];

    groupData[lidy][lidx] = matrix[gidy * width + gidx];

    barrier(CLK_LOCAL_MEM_FENCE);

    // lid (x, y): (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    // result (x, y): (gidy, gidx), (gidy + 1, gidx - 1), (gidy + 2, gidx - 2)
    // у результата получаются соседние x, и одинаковые y т.к. y = gidx - lidx всегда дает одно и то же у соседних потоков (lidx и gidx внутри рабочей группы растут одинаково)
    // при этом глобальные индексы мы используем как опору

    const int resultIdx = gidy - lidy + lidx;
    const int resultIdy = gidx - lidx + lidy;

    result[resultIdy * width + resultIdx] = groupData[lidx][lidy];
}

__kernel void matrix_transpose_local_good_banks_non_coalesced(__global float* matrix, __global float* result, const unsigned int width, const unsigned int height)
{
    const unsigned int gidx = get_global_id(0);
    const unsigned int gidy = get_global_id(1);
    const unsigned int lidx = get_local_id(0);
    const unsigned int lidy = get_local_id(1);

    __local float groupData[SUBMATRIX_SIZE][SUBMATRIX_SIZE];

    const unsigned int stairIndex = (lidx + lidy) % SUBMATRIX_SIZE;
    groupData[lidy][stairIndex] = matrix[gidy * width + gidx];

    barrier(CLK_LOCAL_MEM_FENCE);

    result[gidx * height + gidy] = groupData[lidy][stairIndex];
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

    const int resultIdx = gidy - lidy + lidx;
    const int resultIdy = gidx - lidx + lidy;

    result[resultIdy * width + resultIdx] = groupData[lidx][stairIndex];
}
