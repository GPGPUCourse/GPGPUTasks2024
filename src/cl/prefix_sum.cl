__kernel void prefix_sum_efficient_first(__global unsigned int* as, unsigned int n, unsigned int step)
{
	unsigned int idx = get_global_id(0);
	unsigned int second = (idx + 1) * step * 2 - 1;
	unsigned int first = second - step;
	if (second < n)
		as[second] = as[first] + as[second];
}


__kernel void prefix_sum_efficient_second(__global unsigned int* as, unsigned int n, unsigned int step)
{
	unsigned int idx = get_global_id(0);
	unsigned int second = (idx + 1) * step * 2 - 1 + step;
	unsigned int first = second - step;
	if (second < n)
		as[second] = as[first] + as[second];
}
