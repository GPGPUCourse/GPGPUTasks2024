#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

__kernel void nbody_calculate_force_global(
    __global float * pxs, __global float * pys,
    __global float *vxs, __global float *vys,
    __global const float *mxs,
    __global float * dvx2d, __global float * dvy2d,
    int N,
    int t)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];

    float dvx_sum = 0.f;
    float dvy_sum = 0.f;
    for (unsigned int j = 0; j < N; j++) {
        float x1 = pxs[j];
        float y1 = pys[j];
        float m1 = mxs[j];

        float dx = x1 - x0;
        float dy = y1 - y0;
        float dxy = sqrt(dx * dx + dy * dy);

        *dvx_sum += (G * m0 * m1 / (dxy * dxy * dxy)) * dx;
        *dvy_sum += (G * m0 * m1 / (dxy * dxy * dxy)) * dy;
    }

    *dvx = dvx_sum;
    *dvy = dvy_sum;
}

__kernel void nbody_integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];
}
