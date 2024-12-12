#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

__kernel void nbody_calculate_force_global(
    __global float *pxs, __global float *pys,
    __global float *vxs, __global float *vys,
    __global const float *mxs,
    __global float *dvx2d, __global float *dvy2d,
    unsigned int N,
    int t)
{
    unsigned int gidx = get_global_id(0);

    if (gidx >= N)
        return;

    __global float *dvx = dvx2d + t * N;
    __global float *dvy = dvy2d + t * N;

    float x0 = pxs[gidx];
    float y0 = pys[gidx];
    float m0 = mxs[gidx];

    for (unsigned int i = 0; i < N; i++)
    {
        if (i == gidx)
            continue;
        float x1 = pxs[i];
        float y1 = pys[i];
        float m1 = mxs[i];

        float dx = x1 - x0;
        float dy = y1 - y0;

        float dist_squared = max(100.f, dx * dx + dy * dy);
        float inv_dist = sqrt(1.f / dist_squared);

        float xproj = dx * inv_dist;
        float yproj = dy * inv_dist;

        float fx = xproj / dist_squared * GRAVITATIONAL_FORCE;
        float fy = yproj / dist_squared * GRAVITATIONAL_FORCE;

        dvx[gidx] += m1 * fx;
        dvy[gidx] += m1 * fy;
    }
}

__kernel void nbody_integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        unsigned int N,
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
