#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define INT_MAX 2147483647
#define INT_MIN -2147483648

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

int common_prefix(morton_t a, morton_t b){
    morton_t val = a ^ b;
//    return clz(val);
//     __clz?
    for (int i = NBITS - 1; i >= 0; i--) {
        if(getBit(val, i) == 1) return NBITS - (i + 1);
    }
    return NBITS;
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

    morton_t morton_code = spreadBits(x) | (spreadBits(y) << 1);

    // augmentation
    return (morton_code << 32u) | i;

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


__kernel void buidLBVH(__global const float *pxs, __global const float *pys, __global const float *mxs,
                       __global const morton_t *codes, __global struct Node *nodes,
                       int N)
{
    const int gidx = get_global_id(0);
    if (gidx >= 2 * N - 1)
        return;

    __global struct Node *node = &nodes[gidx];
    clear(&node->bbox);
    node->cmsx = 0;
    node->cmsy = 0;
    node->mass = 0;

    if (gidx >= N - 1){
        int node_idx = gidx - (N - 1);
        node->child_left = -1;
        node->child_right = -1;
        int gt_idx = getIndex(codes[node_idx]);
        float x = pxs[gt_idx];
        float y = pys[gt_idx];
        float m = mxs[gt_idx];
        growPoint(&node->bbox, x, y);
        node->cmsx = x;
        node->cmsy = y;
        node->mass = m;
        return;
    }

    const morton_t curr_code = codes[gidx];
    int i_begin = 0, i_end = N - 1, sign = 1;
    // если рассматриваем не корень, то нужно найти зону ответственности ноды и самый старший бит, с которого надо начинать поиск разреза
    if (gidx > 0) {
        const morton_t prev_code = codes[gidx - 1];
        const morton_t next_code = codes[gidx + 1];
        sign = common_prefix(next_code, curr_code) - common_prefix(curr_code, prev_code);
        sign = max(min(sign, 1), -1);
        const int min_prefix = sign < 0 ? common_prefix(next_code, curr_code) : common_prefix(curr_code, prev_code);
        if (sign > 0){
            int l = gidx, r = N;
            while (r - l > 1){
                int m = (l + r) / 2;
                if(common_prefix(curr_code, codes[m]) > min_prefix) l = m;
                else r = m;
            }
            i_begin = gidx, i_end = l;
        } else if (sign < 0) {
            int l = -1, r = gidx;
            while (r - l > 1){
                int m = (l + r) / 2;
                if (common_prefix(curr_code, codes[m]) > min_prefix) r = m;
                else l = m;
            }
            i_begin = r, i_end = gidx;
        }
    }

    morton_t left_code, right_code;
    if (sign > 0){
        left_code = curr_code, right_code = codes[i_end];
    }else{
        left_code = codes[i_begin], right_code = curr_code;
    }
    const int node_prefix = common_prefix(left_code, right_code);

    int l = i_begin, r = i_end + 1;
    while(r - l > 1){
        int m = (l + r) / 2;
        if(common_prefix(left_code, codes[m]) > node_prefix) l = m;
        else r = m;
    }
    const int split_index = l;
    const int common_left_index = split_index, common_right_index = split_index + 1;

    node->child_left = (common_left_index - i_begin) > 0 ? common_left_index : (N - 1 + common_left_index);
    node->child_right = (i_end - common_right_index) > 0 ? common_right_index : (N - 1 + common_right_index);
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

    float m0 = left->mass;
    float m1 = right->mass;

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

__kernel void calculateForces(
        __global const float *pxs, __global const float *pys,
        __global const float *vxs, __global const float *vys,
        __global const float *mxs,
        __global const struct Node *nodes,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
    int gidx = get_global_id(0);
    if (gidx >= N) return;

    const float x0 = pxs[gidx];
    const float y0 = pys[gidx];

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    int stack[2 * NBITS_PER_DIM];
    int stack_size = 1;
    stack[0] = 0;

    float tmp_dvx = *(dvx + gidx), tmp_dvy = *(dvy + gidx);

    while (stack_size) {
        stack_size--;
        int node_idx = stack[stack_size];
        __global const struct Node *node = &nodes[node_idx];
        if (isLeaf(node)) {
            continue;
        }

        __global const struct Node *left = &nodes[node->child_left];
        __global const struct Node *right = &nodes[node->child_right];

        // если запрос содержится и а левом и в правом ребенке - то они в одном пикселе
        {
            if (contains(&left->bbox, x0, y0) && contains(&right->bbox, x0, y0)) {
                if (!equals(&left->bbox, &right->bbox)) {
                    printf("42357987645432456547\n");
                }
                if (!equalsPoint(&left->bbox, x0, y0)) {
                    printf("5446456456435656\n");
                }
                continue;
            }
        }


        for (int i_child = 0; i_child < 2; ++i_child) {

            __global const struct Node *child = (i_child == 0) ? left : right;
            // С точки зрения ббоксов заходить в ребенка, ббокс которого не пересекаем, не нужно (из-за того, что в листьях у нас точки и они не высовываются за свой регион пространства)
            //   Но, с точки зрения физики, замена гравитационного влияния всех точек в регионе на взаимодействие с суммарной массой в центре масс - это точное решение только в однородном поле (например, на поверхности земли)
            //   У нас поле неоднородное, и такая замена - лишь приближение. Чтобы оно было достаточно точным, будем спускаться внутрь ноды, пока она не станет похожа на точечное тело (маленький размер ее ббокса относительно нашего расстояния до центра масс ноды)
            if (!contains(&child->bbox, x0, y0) && barnesHutCondition(x0, y0, child)) {
                float x1 = child->cmsx;
                float y1 = child->cmsy;
                float m1 = child->mass;

                float dx = x1 - x0;
                float dy = y1 - y0;
                float dr2 = max(100.f, dx * dx + dy * dy);

                float dr2_inv = 1.f / dr2;
                float dr_inv = sqrt(dr2_inv);

                float ex = dx * dr_inv;
                float ey = dy * dr_inv;

                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

                tmp_dvx += m1 * fx;
                tmp_dvy += m1 * fy;
            } else {
                stack[stack_size++] = (i_child == 0) ? node->child_left : node->child_right;
                if (stack_size >= 2 * NBITS_PER_DIM) {
                    printf("0420392384283\n");
                }
            }
        }
    }

    *(dvx + gidx) = tmp_dvx;
    *(dvy + gidx) = tmp_dvy;
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
