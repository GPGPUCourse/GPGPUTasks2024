__kernel void bitonic(__global int *as, unsigned int n, unsigned int k, unsigned int j)
{
    const unsigned int i = get_global_id(0);

    const unsigned int l = i ^ j;

    if (l > i) {
        if ((((i & k) == 0) && (as[i] > as[l])) ||
        (((i & k) != 0) && (as[i] < as[l]))) {
            const int tmp = as[l];
            as[l] = as[i];
            as[i] = tmp;
        }
    }
}
