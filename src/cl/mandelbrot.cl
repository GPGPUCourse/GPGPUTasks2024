#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(volatile __global float* results, unsigned int width, unsigned int height,
                   float fromX, float fromY,
                   float sizeX, float sizeY,
                   unsigned int iters, int smoothing, unsigned int antiAliasing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    if (antiAliasing == 0) {
        printf("Anti aliasing must be greater, than zero!\n");
        return;
    }

    const unsigned int globalIdX = get_global_id(0);
    const unsigned int globalIdY = get_global_id(1);
    const unsigned int i = globalIdX / antiAliasing;
    const unsigned int j = globalIdY / antiAliasing;
    if (i >= height || j >= width)
        return;
    
    float x0 = fromX + (globalIdX + 0.5f * antiAliasing) * sizeX / (width * antiAliasing);
    float y0 = fromY + (globalIdY + 0.5f * antiAliasing) * sizeY / (height * antiAliasing);

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
    result /= (antiAliasing * antiAliasing);
    float old = result;
    volatile __global float* const addres = &results[j * width + i];
    while ((old = atomic_xchg(addres, atomic_xchg(addres, -1.f) + old) != -1.f));
}
