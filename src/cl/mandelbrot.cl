#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(const float sizeX,
                         const float sizeY,
                         const unsigned int width,
                         const unsigned int height,
                         const float fromX,
                         const float fromY,
                         const unsigned int iters,
                         unsigned int smoothing, // а где bool?
                         __global float* results)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    if (idx >= width || idy >= height)
        return;

    float x0 = fromX + (idx + 0.5f) * sizeX / width;  // как же я пожалел, что впихнул сюда тернарный оператор...
    float y0 = fromY + (idy + 0.5f) * sizeY / height;

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
    results[idy * width + idx] = result;
}
