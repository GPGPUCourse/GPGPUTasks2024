__kernel void bitonic(__global int *as, const unsigned int i, const unsigned int j) {
    unsigned int idx = get_global_id(0);

    unsigned int ixj = idx ^ (j / 2);

    if (ixj > idx) {
        int direction = (idx & i) == 0;

        if ((as[idx] > as[ixj]) == direction) {
            int temp = as[idx];
            as[idx] = as[ixj];
            as[ixj] = temp;
        }
    }
}
