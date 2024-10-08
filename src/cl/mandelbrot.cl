#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

float recalc_point(float x0, float y0, unsigned int iters, const float threshold, const float threshold2, int smoothing) {
    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;

    return result;
}

__kernel void mandelbrot(
    __global float* results,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, int smoothing,
    unsigned int antialiasingLevel
)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= width || j >= height) {
        return;
    }

    float result = 0;
    for (int ii = 0; ii <= antialiasingLevel; ++ii) {
        for (int jj = 0; jj <= antialiasingLevel; ++jj) {
            float x0 = fromX + (i + (1.0f * (ii + 1) / (antialiasingLevel + 2))) * sizeX / width;
            float y0 = fromY + (j + (1.0f * (jj + 1) / (antialiasingLevel + 2))) * sizeY / height;
            result += recalc_point(x0, y0, iters, threshold, threshold2, smoothing);
        }
    }

    result /= (antialiasingLevel + 1) * (antialiasingLevel + 1);
    results[j * width + i] = result;
}
