__kernel void bitonic(__global int *data, const unsigned int stage, const unsigned int passOfStage)
{
    const unsigned int gidx = get_global_id(0);

    unsigned int internalIdx;
    unsigned int secondIdx;

    if (passOfStage > 1) {
        internalIdx = (gidx / (passOfStage / 2)) * passOfStage + (gidx % (passOfStage / 2));
        secondIdx = internalIdx + (passOfStage / 2);
    }
    else {
        internalIdx = gidx;
        secondIdx = internalIdx + 1;
    }

    int first = data[internalIdx];
    int second = data[secondIdx];

    int direction = ((gidx / (stage / 2)) % 2) == 0 ? 1 : 0; // 1 - возрастающая, 0 - убывающая

    if ( (direction && first > second) || (!direction && first < second) ) {
        data[internalIdx] = second;
        data[secondIdx] = first;
    }
}
