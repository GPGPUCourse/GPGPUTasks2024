__kernel void bitonic(__global int* data, unsigned int i, unsigned int j) {
    unsigned int gid = get_global_id(0);
    unsigned int k = gid ^ j;

    if (k > gid) {
        int left = data[gid];
        int right = data[k];

        bool isAscendingDirection = ((gid & i) == 0);
        if ((left > right) == isAscendingDirection) {
            data[gid] = right;
            data[k] = left;
        }
    }
}
