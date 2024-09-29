#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define THRESHOLD (256.0f)
#define THRESHOLD_SQUARED (THRESHOLD * THRESHOLD)

float calculateForX0Y0(
    float x0, float y0,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, unsigned int smoothing
) {
    float x = x0;
    float y = y0;

    unsigned int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > THRESHOLD_SQUARED) {
            break;
        }
    }

    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log2(log2(sqrt(x * x + y * y)) / log2(THRESHOLD)) / log2(2.0f);
    }

    return (1.0f * result / iters);
}

__kernel void mandelbrot(
    __global float* results,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, unsigned int smoothing,
    unsigned int antialiasing_level
)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= width || j >= height) {
        return;
    }

    if (antialiasing_level <= 1) {
        float x0 = fromX + (i + 0.5f) * sizeX / width;
        float y0 = fromY + (j + 0.5f) * sizeY / height;
        results[j * width + i] = calculateForX0Y0(x0, y0, width, height, fromX, fromY, sizeX, sizeY, iters, smoothing);
    }
    else {
        // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
        // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
        // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
        // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
        
        /*
        Antialiasing Level -> Step
        2 -> 1.0f
        3 -> 0.5f
        4 -> 0.33333f
        etc.
        */
        float step = 1.0f / (antialiasing_level - 1);

        for (float step_y = 0.0f; step_y <= 1.0f; step_y += step) {
            for (float step_x = 0.0f; step_x <= 1.0f; step_x += step) {
                float x0 = fromX + (i + step_x) * sizeX / width;
                float y0 = fromY + (j + step_y) * sizeY / height;

                results[j * width + i] += calculateForX0Y0(x0, y0, width, height, fromX, fromY, sizeX, sizeY, iters, smoothing);
            }
        }

        results[j * width + i] /= (antialiasing_level * antialiasing_level);
    }
}
