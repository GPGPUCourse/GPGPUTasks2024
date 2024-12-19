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

    float dvx_i = 0.0f;
    float dvy_i = 0.0f;

    for (int j = 0; j < N; ++j) {
        if (j == i) {
            continue;
        }

        float x1 = pxs[j];
        float y1 = pys[j];
        float m1 = mxs[j];

        float dx = x1 - x0;
        float dy = y1 - y0;

        float dr2 = dx * dx + dy * dy;
        dr2 = max(100.0f, dr2);

        float dr_inv = sqrt(1.0f / dr2);
        float ex = dx * dr_inv;
        float ey = dy * dr_inv;

        float fx = ex * (1.0f / dr2) * GRAVITATIONAL_FORCE;
        float fy = ey * (1.0f / dr2) * GRAVITATIONAL_FORCE;

        dvx_i += m1 * fx;
        dvy_i += m1 * fy;
    }

    dvx[i] = dvx_i;
    dvy[i] = dvy_i;
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
