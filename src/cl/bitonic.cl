__kernel void bitonic(__global int* as, unsigned int max_K, unsigned int K, unsigned int n)
{
	
	unsigned int gidx = get_global_id(0);
	unsigned int block_size = 2 * K;
	unsigned int block_idx = gidx / K;
	
	unsigned int arrow_in_block = gidx % K;
	
	unsigned int first = block_idx * block_size + arrow_in_block;
	unsigned int second = first + K;
	if (second >= n || first >= n)
		return;
	
	int first_value = as[first];
	int second_value = as[second];
	
	bool asc = ((gidx / max_K) % 2) == 0;
	
	if (first_value > second_value && asc)
	{
		as[second] = first_value;
		as[first] = second_value;
	}

	if (first_value < second_value && !asc)
	{
		as[second] = first_value;
		as[first] = second_value;
	}
	
}
