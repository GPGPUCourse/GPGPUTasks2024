#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters, int smoothing)
{
    // если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    int i = get_global_id(0),
        j = get_global_id(1);

    if (i >= width || j >= height) return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float result = 0.0;

#ifdef ANTI_ALIASING_ENABLED
    const unsigned int N = 4;
    for (int offX = 0; offX <= N; ++offX) {
        float x0 = fromX + (i + (float)offX / N) * sizeX / width;
        for (int offY = 0; offY <= N; ++offY) {
            float y0 = fromY + (j + (float)offY / N) * sizeY / height;
#else
    const unsigned int N = 0;
    {
        float x0 = fromX + (i + 0.5) * sizeX / width;
        {
            float y0 = fromY + (j + 0.5) * sizeY / height;
#endif
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
            float res = iter;
            if (smoothing && iter != iters) {
                res = res - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
            }
            result += res;
        }
    }
    result /= (N + 1) * (N + 1) * iters;
    results[j * width + i] = result;
}
