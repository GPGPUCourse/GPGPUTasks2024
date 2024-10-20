__kernel void bitonic(__global int* data, unsigned int n_reds, unsigned int red_i) {
    unsigned int gid = get_global_id(0);
    bool ascending = (gid / (1 << n_reds)) % 2;
    int jump = 1 << (n_reds - red_i - 1);
    bool active = !((gid / jump) % 2);


    if (active) {
        if ((data[gid] > data[gid + jump]) ^ ascending) {
            int temp = data[gid];
            data[gid] = data[gid + jump];
            data[gid + jump] = temp;
        }
    }
}