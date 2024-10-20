__kernel void bitonic(__global int *as, unsigned int n, unsigned int j, unsigned int k) {
    unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    unsigned int l = i ^ j;
    if (l > i) {
        if ((!(i & k) && as[i] > as[l]) || ((i & k) && as[i] < as[l])) {
            unsigned int tmp = as[l];
            as[l] = as[i];
            as[i] = tmp;
        }
    }
}

