__kernel void bitonic(__global int *as, const unsigned int big_size, const unsigned int small_size)
{
    int gid = get_global_id(0);
    int pairIdx = gid ^ (small_size / 2);
    if (pairIdx > gid) {
        int dir = ((gid & big_size) == 0);
        int elem1 = as[gid];
        int elem2 = as[pairIdx];
        if ((elem1 > elem2) == dir) {
            as[gid] = elem2;
            as[pairIdx] = elem1;
        }
    }
}

