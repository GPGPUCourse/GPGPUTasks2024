// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 6;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

// exponential smooth min (k=32)
float smin(float a, float b, float k)
{
    float res = exp2(-k * a) + exp2(-k * b);
    return -log2(res) / k;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    float right_hand = sdCapsule(p - vec3(-0.28, 0.35, -0.7), vec3(-0.1, -0.2 * lazycos(5.0 * iTime), 0.2), vec3(0), 0.04);
    float left_hand  = sdCapsule(p - vec3( 0.28, 0.35, -0.7), vec3( 0.1, -0.2,                        0.2), vec3(0), 0.04);
    float hands = smin(left_hand, right_hand, 32.0);

    float left_leg  = sdSphere(p - vec3(-0.2, 0.02, -0.7), 0.045);
    float right_leg = sdSphere(p - vec3( 0.2, 0.02, -0.7), 0.045);
    float legs = smin(left_leg, right_leg, 32.0);

    float body = sdSphere(p - vec3(0.0, 0.35, -0.7), 0.3);
    body = smin(body, hands, 32.0);
    body = smin(legs, body,  32.0);

    return vec4(body, vec3(0, 1, 0));
}

vec4 sdEye(vec3 p)
{
    vec3 center = p - vec3(0.0, 0.45, -0.53);
    float eye = sdSphere(center, 0.15);
    float white_part = sdSphere(center - vec3(0.0, 0.025, 0.1),  0.08);
    float black_part = sdSphere(center - vec3(0.0, 0.025, 0.16), 0.04);
    if (eye < white_part) {
        return vec4(eye, 1, 1, 1);
    } else if (white_part < black_part) {
        return vec4(white_part, 0, 1, 1);
    } else {
        return vec4(black_part, 0, 0, 0);
    }
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    return res;
}

vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);

    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal(in vec3 p)// for function f(p)
{
    const float eps = 0.0001;// or some other value
    const vec2 h = vec2(eps, 0);
    return normalize(
        vec3(
            sdTotal(p + h.xyy).x - sdTotal(p - h.xyy).x,
            sdTotal(p + h.yxy).x - sdTotal(p - h.yxy).x,
            sdTotal(p + h.yyx).x - sdTotal(p - h.yyx).x
        )
    );
}

vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    const float EPS = 1e-3;
    float t = 0.0;

    for (int iter = 0; iter < 200; iter++) {
        vec4 res = sdTotal(ray_origin + t * ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
}

float shading(vec3 p, vec3 light_source, vec3 normal)
{
    vec3 light_dir = normalize(light_source - p);
    float shading = dot(light_dir, normal);
    return clamp(shading, 0.5, 1.0);
}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shininess)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shininess);
}

float castShadow(vec3 p, vec3 light_source)
{
    vec3 light_dir = p - light_source;
    float target_dist = length(light_dir);

    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5 * wh, -1.0));

    vec4 res = raycast(ray_origin, ray_direction);

    vec3 col = res.yzw;

    vec3 surface_point = ray_origin + res.x * ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5 * sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(spec);

    // Output to screen
    fragColor = vec4(col, 1.0);
}

