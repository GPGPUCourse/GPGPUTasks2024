#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define ANTIALIASING 1

float compute(float x0, float y0, unsigned int max_iter, unsigned int smoothing) {
    const float threshold = 256.0;
    const float threshold2 = threshold * threshold;
    const float inv_log_threshold2 = 1.0 / (2.0 * log(threshold));
    const float inv_log_two = 1.0 / log(2.0);

    float x = 0.0;
    float y = 0.0;

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        float xp = x;
        x = x * x - y * y + x0;
        float xp_y = xp * y;
        y = xp_y + xp_y + y0;
        if (x * x + y * y > threshold2) {
            break;
        }
    }
    float res = iter;
    if (smoothing && iter < max_iter) {
        res -= log(log(x * x + y * y) * inv_log_threshold2) * inv_log_two;
    }

    return res / max_iter;
}

__kernel void mandelbrot(
        __global float *result,
        unsigned int width,
        unsigned int height,
        float x0,
        float y0,
        float size_x,
        float size_y,
        unsigned int max_iter,
        unsigned int smoothing
) {
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const float threshold = 256.0;
    const float threshold2 = threshold * threshold;
    const float inv_log_threshold2 = 1.0 / (2.0 * log(threshold));
    const float inv_log_two = 1.0 / log(2.0);

    int j = get_global_id(0);
    int i = get_global_id(1);

    float sum = 0.0;

    for (int aa_i = 1; aa_i <= ANTIALIASING; ++aa_i) {
        for (int aa_j = 1; aa_j <= ANTIALIASING; ++aa_j) {
            float px0 = x0 + size_x * (((float)j + (float)aa_i / (ANTIALIASING + 1)) / width);
            float py0 = y0 + size_y * (((float)i + (float)aa_j / (ANTIALIASING + 1)) / height);
            sum += compute(px0, py0, max_iter, smoothing);
        }
    }

    result[i * width + j] = sum / (ANTIALIASING * ANTIALIASING);
}
