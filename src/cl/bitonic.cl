
__kernel void bitonic(
    __global int *as, 
    const unsigned int compare_distance,
    const unsigned int partition_size
) {
    unsigned int array_index = get_global_id(0);
    unsigned int compare_distance_array_index = compare_distance ^ array_index;
    
    // для избежания повторных перестановок
    if (compare_distance_array_index > array_index) {
        unsigned int sequence_direction = array_index & partition_size;
        
        bool swap = (!sequence_direction && as[array_index] > as[compare_distance_array_index]) ||
                     (sequence_direction && as[array_index] < as[compare_distance_array_index]);
        
        if (swap) {
            int temp_var = as[array_index];
            as[array_index] = as[compare_distance_array_index];
            as[compare_distance_array_index] = temp_var;
        }
    }
}
