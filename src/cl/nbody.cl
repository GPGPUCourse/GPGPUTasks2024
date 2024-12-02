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

    dvx[i] = 0;
    dvy[i] = 0;
    for (int k = 1; k < N; ++k) {
        int j = i + k;
        if (j >= N) j -= N;

        float dx = pxs[j] - x0;
        float dy = pys[j] - y0;
        float dst2 = max(100.0f, dx * dx + dy * dy);
        float dst3 = sqrt(dst2) * dst2;
        float scl = mxs[j] / dst3;
        dvx[i] += scl * dx;
        dvy[i] += scl * dy;
    }
    dvx[i] *= GRAVITATIONAL_FORCE;
    dvy[i] *= GRAVITATIONAL_FORCE;
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
