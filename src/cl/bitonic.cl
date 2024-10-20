__kernel void bitonic(
    __global int* as,
    unsigned int as_size,
    unsigned int sort_size,
    unsigned int block_size
)
{
    const unsigned int i = get_global_id(0);

    const unsigned int half_sort_size = sort_size / 2;
    const unsigned int half_block_size = block_size / 2;

    const unsigned int sort_i = i / half_sort_size;
    const int descending = sort_i % 2;

    const unsigned int block_i = i / half_block_size;
    const unsigned int i_in_block = i % half_block_size;

    const unsigned int ii = block_i * block_size + i_in_block;

    const int l = as[ii];
    const int r = as[ii + half_block_size];

    as[ii] = select(l, r, (r < l) ^ descending);
    as[ii + half_block_size] = select(r, l, (r < l) ^ descending);
}
