#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void bitonic(__global int *as, const unsigned int outerblockSize, const unsigned int innerBlockSize)
{
    const unsigned int gidx = get_global_id(0);

    unsigned int internalIdx = (gidx / (innerBlockSize / 2)) * innerBlockSize + gidx % (innerBlockSize / 2);

    int first = as[internalIdx];
    int second = as[internalIdx + innerBlockSize / 2];

    if ((first > second && (gidx / (outerblockSize / 2)) % 2 == 0) ||
        (first < second && (gidx / (outerblockSize / 2)) % 2 == 1))
    {
        as[internalIdx] = second;
        as[internalIdx + innerBlockSize / 2] = first;
    }
}
