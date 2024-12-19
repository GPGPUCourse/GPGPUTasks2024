#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


#define GRAVITATIONAL_FORCE 0.0001

#define morton_t ulong

#define NBITS_PER_DIM 16
#define NBITS (NBITS_PER_DIM /*x dimension*/ + NBITS_PER_DIM /*y dimension*/ + 32 /*index augmentation*/)

int LBVHSize(int N) {
    return N + N-1;
}

morton_t getBits(morton_t morton_code, int bit_index, int prefix_size)
{
    morton_t one = 1;
    return (morton_code >> bit_index) & ((one << prefix_size) - one);
}

int getBit(morton_t morton_code, int bit_index)
{
    return (morton_code >> bit_index) & 1;
}

int getIndex(morton_t morton_code)
{
    morton_t mask = 1;
    mask = (mask << 32) - 1;
    return morton_code & mask;
}

int spreadBits(int word){
    word = (word ^ (word << 8 )) & 0x00ff00ff;
    word = (word ^ (word << 4 )) & 0x0f0f0f0f;
    word = (word ^ (word << 2 )) & 0x33333333;
    word = (word ^ (word << 1 )) & 0x55555555;
    return word;
}

struct __attribute__ ((packed)) BBox {

    int minx, maxx;
    int miny, maxy;

};

void clear(__global struct BBox *self)
{
    self->minx = INT_MAX;
    self->maxx = INT_MIN;
    self->miny = self->minx;
    self->maxy = self->maxx;
}

bool contains(__global const struct BBox *self, float fx, float fy)
{
    int x = fx + 0.5;
    int y = fy + 0.5;
    return x >= self->minx && x <= self->maxx &&
           y >= self->miny && y <= self->maxy;
}

bool empty(__global const struct BBox *self)
{
    return self->minx > self->maxx;
}

struct __attribute__ ((packed)) Node {

    int child_left, child_right;
    struct BBox bbox;

    // used only for nbody
    float mass;
    float cmsx;
    float cmsy;
};

bool hasLeftChild(__global const struct Node *self)
{
    return self->child_left >= 0;
}

bool hasRightChild(__global const struct Node *self)
{
    return self->child_right >= 0;
}

bool isLeaf(__global const struct Node *self)
{
    return !hasLeftChild(self) && !hasRightChild(self);
}

void growPoint(__global struct BBox *self, float fx, float fy)
{
    self->minx = min(self->minx, (int) (fx + 0.5));
    self->maxx = max(self->maxx, (int) (fx + 0.5));
    self->miny = min(self->miny, (int) (fy + 0.5));
    self->maxy = max(self->maxy, (int) (fy + 0.5));
}

void growBBox(__global struct BBox *self, __global const struct BBox *other)
{
    growPoint(self, other->minx, other->miny);
    growPoint(self, other->maxx, other->maxy);
}

bool equals(__global const struct BBox *lhs, __global const struct BBox *rhs)
{
    return lhs->minx == rhs->minx && lhs->maxx == rhs->maxx && lhs->miny == rhs->miny && lhs->maxy == rhs->maxy;
}

bool equalsPoint(__global const struct BBox *lhs, float fx, float fy)
{
    int x = fx + 0.5;
    int y = fy + 0.5;
    return lhs->minx == x && lhs->maxx == x && lhs->miny == y && lhs->maxy == y;
}

morton_t zOrder(float fx, float fy, int i){
    int x = fx + 0.5;
    int y = fy + 0.5;

    // у нас нет эксепшенов, но можно писать коды ошибок просто в консоль, и следить чтобы вывод был пустой

    if (x < 0 || x >= (1 << NBITS_PER_DIM)) {
        printf("098245490432590890\n");
//        return 0;
    }
    if (y < 0 || y >= (1 << NBITS_PER_DIM)) {
        printf("432764328764237823\n");
//        return 0;
    }

    int mx = spreadBits(x);
    int my = spreadBits(y);
    ulong morton_code = ((ulong)mx) | (((ulong)my) << 1);
    return (morton_code << 32) | (ulong)i;
}

__kernel void generateMortonCodes(__global const float *pxs, __global const float *pys,
                                  __global morton_t *codes,
                                  int N)
{
    int gid = get_global_id(0);
    if (gid >= N)
        return;

    codes[gid] = zOrder(pxs[gid], pys[gid], gid);
}

bool mergePathPredicate(morton_t val_mid, morton_t val_cur, bool is_right)
{
    return is_right ? val_mid <= val_cur : val_mid < val_cur;
}

void __kernel merge(__global const morton_t *as, __global morton_t *as_sorted, unsigned int n, unsigned int subarray_size)
{
    const int gid = get_global_id(0);
    if (gid >= n)
        return;

    const int subarray_id = gid / subarray_size;
    const int is_right_subarray = subarray_id & 1;

    const int base_cur = (subarray_id) * subarray_size;
    const int base_other = (subarray_id + 1 - 2 * is_right_subarray) * subarray_size;

    const int j = gid - base_cur;
    const morton_t val_cur = as[gid];

    int i0 = -1;
    int i1 = subarray_size;
    while (i1 - i0 > 1) {
        int mid = (i0 + i1) / 2;
        if (base_other + mid < n && mergePathPredicate(as[base_other + mid], val_cur, is_right_subarray)) {
            i0 = mid;
        } else {
            i1 = mid;
        }
    }
    const int i = i1;

    int idx = min(base_cur, base_other) + j + i;
    as_sorted[idx] = val_cur;
}

int findSplit(__global const morton_t *codes, int i_begin, int i_end, int bit_index)
{
    int start_bit = getBit(codes[i_begin], bit_index);
    int end_bit = getBit(codes[i_end - 1], bit_index);

    if (start_bit == end_bit) {
        return -1;
    }

    int low = i_begin + 1;
    int high = i_end;
    int first_split = -1;
    while (low < high) {
        int mid = (low + high) / 2;
        int bit = getBit(codes[mid], bit_index);
        if (bit != start_bit) {
            first_split = mid;
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return first_split;
}

void findRegion(int *i_begin, int *i_end, int *bit_index, __global const morton_t *codes, int N, int i_node)
{
        // This logic is based on the CPU version in main_lbvh.cpp
    // findRegion tries to find subregion of codes corresponding to node i_node

    // i_node in [1, N-2] for internal nodes
    if (i_node < 1 || i_node > N - 2) {
        //printf("842384298293482\n");
    }

    int found_bit = -1;
    int dir = 0;
    // We try to guess direction and bit where codes differ
    for (int b = NBITS - 1; b >= 0; --b) {
        int l = getBit(codes[i_node - 1], b);
        int m = getBit(codes[i_node], b);
        int r = getBit(codes[i_node + 1], b);
        if (l == 0 && m == 0 && r == 1) {
            dir = -1;
            found_bit = b;
            break;
        }

        if (l == 0 && m == 1 && r == 1) {
            dir = 1;
            found_bit = b;
            break;
        }
    }

    if (dir == 0) {
        //printf("8923482374983\n");
    }

    int i_bit = found_bit;
    int K = NBITS - i_bit;
    morton_t pref0 = getBits(codes[i_node], i_bit, K);

    *bit_index = i_bit;

    if (dir > 0) {
        int low = i_node;
        int high = N;
        while (low != high) {
            int mid = (low + high) / 2;
            if (getBits(codes[mid], i_bit, K) == pref0) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        *i_begin = i_node;
        *i_end = low;
    } else {
        int low = 0;
        int high = i_node;
        while (low != high) {
            int mid = (low + high) / 2;
            if (!(getBits(codes[mid], i_bit, K) == pref0)) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        *i_begin = low;
        *i_end = i_node + 1;
    }
}


void initLBVHNode(__global struct Node *nodes, int i_node, __global const morton_t *codes, int N, __global const float *pxs, __global const float *pys, __global const float *mxs)
{
        clear(&nodes[i_node].bbox);
    nodes[i_node].mass = 0.f;
    nodes[i_node].cmsx = 0.f;
    nodes[i_node].cmsy = 0.f;

    if (i_node >= N - 1) {
        // leaf
        nodes[i_node].child_left = -1;
        nodes[i_node].child_right = -1;
        int i_point = i_node - (N - 1);
        int idx = getIndex(codes[i_point]);

        float cx = pxs[idx];
        float cy = pys[idx];
        float mass = mxs[idx];

        growPoint(&nodes[i_node].bbox, cx, cy);
        nodes[i_node].cmsx = cx;
        nodes[i_node].cmsy = cy;
        nodes[i_node].mass = mass;
        return;
    }

    int i_begin = 0, i_end = N, bit_index = NBITS - 1;
    if (i_node > 0) {
        findRegion(&i_begin, &i_end, &bit_index, codes, N, i_node);
    }

    int found = 0;
    for (int b = bit_index; b >= 0; --b) {
        int split = findSplit(codes, i_begin, i_end, b);
        if (split < 0)
            continue;

        int left_size = split - i_begin;
        int right_size = i_end - split;

        if (left_size == 1) {
            nodes[i_node].child_left = (N - 1) + i_begin;
        } else {
            nodes[i_node].child_left = (split - 1);
        }

        if (right_size == 1) {
            nodes[i_node].child_right = (N - 1) + i_end - 1;
        } else {
            // Use the exact same logic as CPU:
            nodes[i_node].child_right = split;
        }

        found = 1;
        break;
    }

    if (!found) {
        //printf("54356549645\n");
    }
}

__kernel void buidLBVH(__global const float *pxs, __global const float *pys, __global const float *mxs,
                       __global const morton_t *codes, __global struct Node *nodes,
                       int N)
{
    int gid = get_global_id(0);
    if (gid >= LBVHSize(N))
        return;

    initLBVHNode(nodes, gid, codes, N, pxs, pys, mxs);
}

void initFlag(__global int *flags, int i_node, __global const struct Node *nodes, int level)
{
    flags[i_node] = -1;

    __global const struct Node *node = &nodes[i_node];
    if (isLeaf(node)) {
        printf("9423584385834\n");
//        return;
    }

    if (!empty(&node->bbox)) {
        return;
    }

    __global const struct BBox *left = &nodes[node->child_left].bbox;
    __global const struct BBox *right = &nodes[node->child_right].bbox;

    if (!empty(left) && !empty(right)) {
        flags[i_node] = level;
    }
}

__kernel void initFlags(__global int *flags, __global const struct Node *nodes,
                       int N, int level)
{
    int gid = get_global_id(0);

    if (gid == N-1)
        flags[gid] = 0; // use last element as a n_updated counter in next kernel

    if (gid >= N-1) // инициализируем только внутренние ноды
        return;

    initFlag(flags, gid, nodes, level);
}

void growNode(__global struct Node *root, __global struct Node *nodes)
{
    __global const struct Node *left = &nodes[root->child_left];
    __global const struct Node *right = &nodes[root->child_right];

    growBBox(&root->bbox, &left->bbox);
    growBBox(&root->bbox, &right->bbox);

    double m0 = left->mass;
    double m1 = right->mass;

    root->mass = m0 + m1;

    if (root->mass <= 1e-8) {
        printf("04230420340322\n");
//        return;
    }

    root->cmsx = (left->cmsx * m0 + right->cmsx * m1) / root->mass;
    root->cmsy = (left->cmsy * m0 + right->cmsy * m1) / root->mass;
}

__kernel void growNodes(__global int *flags, __global struct Node *nodes,
                        int N, int level)
{
    int gid = get_global_id(0);

    if (gid >= N-1) // инициализируем только внутренние ноды
        return;

    __global struct Node *node = &nodes[gid];
    if (flags[gid] == level) {
        growNode(node, nodes);
        atomic_add(&flags[N-1], 1);
    }
}

// https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
bool barnesHutCondition(float x, float y, __global const struct Node *node)
{
    float dx = x - node->cmsx;
    float dy = y - node->cmsy;
    float s = max(node->bbox.maxx - node->bbox.minx, node->bbox.maxy - node->bbox.miny);
    float d2 = dx*dx + dy*dy;
    float thresh = 0.5;

    return s * s < d2 * thresh * thresh;
}

void calculateForce(float x0, float y0, float m0, __global const struct Node *nodes, __global float *force_x, __global float *force_y)
{
        // stack-based approach similar to CPU
    int stack[2 * NBITS_PER_DIM];
    int stack_size = 0;

    stack[stack_size++] = 0; // root

    while (stack_size > 0) {
        int i_node = stack[--stack_size];
        __global const struct Node *node = &nodes[i_node];

        if (isLeaf(node)) {
            float dx = node->cmsx - x0;
            float dy = node->cmsy - y0;
            float dr2 = fmax(100.0f, dx*dx + dy*dy);
            float dr2_inv = 1.0f / dr2;
            float dr_inv = sqrt(dr2_inv);
            float ex = dx * dr_inv;
            float ey = dy * dr_inv;
            float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
            float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;
            *force_x += node->mass * fx;
            *force_y += node->mass * fy;
            continue;
        }

        __global const struct Node *left = &nodes[node->child_left];
        __global const struct Node *right = &nodes[node->child_right];

        bool left_contains = contains(&left->bbox, x0, y0);
        bool right_contains = contains(&right->bbox, x0, y0);

        if (left_contains && right_contains) {
            // if both contain point
            if (!equals(&left->bbox, &right->bbox)) {
                //printf("42357987645432456547\n");
            }
            if (!equalsPoint(&left->bbox, x0, y0)) {
                //printf("5446456456435656\n");
            }
            // no force contribution
            continue;
        }

        // If barnesHut condition is met, treat node as a whole
        // Otherwise, go deeper
        for (int i_child = 0; i_child < 2; i_child++) {
            int c = (i_child == 0) ? node->child_left : node->child_right;
            __global const struct Node *child = &nodes[c];

            bool child_contains_point = contains(&child->bbox, x0, y0);

            if (!child_contains_point && barnesHutCondition(x0, y0, child)) {
                float dx = child->cmsx - x0;
                float dy = child->cmsy - y0;
                float dr2 = fmax(100.f, dx*dx + dy*dy);
                float dr2_inv = 1.f / dr2;
                float dr_inv = sqrt(dr2_inv);
                float ex = dx * dr_inv;
                float ey = dy * dr_inv;
                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;
                *force_x += child->mass * fx;
                *force_y += child->mass * fy;
            } else {
                if (stack_size < 32) {
                    stack[stack_size++] = c;
                } else {
                    //printf("0420392384283\n");
                }
            }
        }
    }
}

__kernel void calculateForces(
        __global const float *pxs, __global const float *pys,
        __global const float *vxs, __global const float *vys,
        __global const float *mxs,
        __global const struct Node *nodes,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
    int i = get_global_id(0);
    if (i >= N) return;

    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];

    float fx = 0.f;
    float fy = 0.f;

    __global float *dvx = dvx2d + t*N;
    __global float *dvy = dvy2d + t*N;
    
    calculateForce(x0, y0, m0, nodes, dvx + i, dvy + i);

}

__kernel void integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t,
        int coord_shift)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];

    // отражаем частицы от границ мира, чтобы не ломался подсчет мортоновского кода
    if (pxs[i] < 1) {
        vxs[i] *= -1;
        pxs[i] += vxs[i];
    }
    if (pys[i] < 1) {
        vys[i] *= -1;
        pys[i] += vys[i];
    }
    if (pxs[i] >= 2 * coord_shift - 1) {
        vxs[i] *= -1;
        pxs[i] += vxs[i];
    }
    if (pys[i] >= 2 * coord_shift - 1) {
        vys[i] *= -1;
        pys[i] += vys[i];
    }
}