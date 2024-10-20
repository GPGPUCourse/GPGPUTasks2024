__kernel void bitonic(__global int *arr, 
                    const unsigned int outer, 
                    const unsigned int inner)
{
    const unsigned int gid = get_global_id(0);
    unsigned int gid_pair = gid ^ inner;
    if (gid_pair > gid) {
        unsigned int arr_gid = arr[gid];
        unsigned int arr_gid_pair = arr[gid_pair];
        if ( ((gid & outer) == 0) && (arr[gid] > arr[gid_pair]) || ((gid & outer) != 0) && (arr[gid] < arr[gid_pair]) ) {
            arr[gid] = arr_gid_pair;
            arr[gid_pair] = arr_gid;
        }
    }
}
