#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(
        __global float* out, const unsigned int width, const unsigned int height,
        const float startX, const float startY, const float sizeX, const float sizeY,
        const unsigned int iterationsLimit, const unsigned int aa
        )
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const unsigned int globalId = get_global_id(0);

    if (globalId >= width * height)
        return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int i = globalId % width;
    const unsigned int j = globalId / width;
    const float aa_factor = aa * aa;

    float result = 0;
    for (int ii = 0; ii < aa; ++ii){
        for (int jj = 0; jj < aa; ++jj) {
            const float x0 = startX + (i + 1.0f * (ii + 1) / (aa + 1)) * sizeX / width;
            const float y0 = startY + (j + 1.0f * (jj + 1) / (aa + 1)) * sizeY / height;

            float x = x0;
            float y = y0;

            int iter = 0;

            for (; iter < iterationsLimit; ++iter) {
                float xPrev = x;
                x = x * x - y * y + x0;
                y = 2.0f * xPrev * y + y0;
                if ((x * x + y * y) > threshold2) {
                    break;
                }
            }

            result += iter / aa_factor;
        }
    }

    out[globalId] = 1.0f * result / iterationsLimit;
}
