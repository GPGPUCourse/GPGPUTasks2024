__kernel void bitonic(__global int *as, unsigned int block_size, unsigned int depth)
{
    const size_t gid = get_global_id(0);
    const size_t sequence_size = block_size >> depth;
    const size_t sequence_index = gid / (sequence_size / 2);
    const size_t sequence_offset = gid % (sequence_size / 2);

    const size_t i1 = sequence_index * sequence_size + sequence_offset;
    const size_t i2 = i1 + sequence_size / 2;

    const int lhs = as[i1];
    const int rhs = as[i2];

    if ((gid / (block_size / 2)) % 2 == 0) {
        if (lhs > rhs) {
            as[i1] = rhs;
            as[i2] = lhs;
        }
    } else {
        if (lhs < rhs) {
            as[i1] = rhs;
            as[i2] = lhs;
        }
    }
}
