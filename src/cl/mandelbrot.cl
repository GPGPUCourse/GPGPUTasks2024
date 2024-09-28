#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(
    __global float *results,
    unsigned int width, unsigned int height,
    float startX, float startY,
    float stepX, float stepY,
    const int maxIter, int smoothing
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const float border1 = 256.0f;
    const float border2 = border1 * border1;
    float x0 = startX + (i + 0.5f) * stepX / width;
    float y0 = startY + (j + 0.5f) * stepY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < maxIter; ++iter) {
        const float xP = x;
        x = x * x - y * y + x0;
        y = 2.0f * xP * y + y0;
        if (x * x + y * y > border2) {
            break;
        }
    }
    float result = iter;
    if (iter != maxIter && smoothing) {
        result = result - log(log(sqrt(x * x + y * y)) / log(border1)) / log(2.0f);
    }

    result *= 1.0f;
    result /= maxIter;
    results[j * width + i] = result;
}
