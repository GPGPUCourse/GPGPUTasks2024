__kernel void bitonic(
    __global int *as,
    unsigned int n,
    unsigned int k,
    unsigned int j
) {
    unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    // source: https://en.wikipedia.org/wiki/Bitonic_sorter
    unsigned int l = i ^ j;
    if (l > i) {
        bool ascending = ((i & k) == 0);
        // printf("i = %d, l = %d, ascending = %d\n", i, l, ascending);
        if ((ascending && as[i] > as[l]) || (!ascending && as[i] < as[l])) {
            unsigned int tmp = as[i];
            as[i] = as[l];
            as[l] = tmp;
        }
    }
}
