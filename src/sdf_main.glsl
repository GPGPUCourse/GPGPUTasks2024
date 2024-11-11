
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

// Capsule / Line - exact
float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

// синус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazysin(float angle, int periods)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < periods) {
        return sin(angle);
    }

    return 0.0;
}

float lazysin(float angle)
{
    return lazysin(angle, 3);
}

// soft min (sigmoid)
float smin(float a, float b, float k)
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

vec4 comb(vec4 a, vec4 b)
{
    return a.x < b.x ? a : b;
}

vec4 sdTree(vec3 p)
{
    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);

    float waveOff = 0.01 * lazysin(5.0 * iTime + 3.0, 1);
    res = comb(res, vec4(sdSphere((p - vec3(0.0 + waveOff, 0.9, 0.0)), 0.3), 0.1, 0.9 + 0.1 * sin(10.0 * (p.x + p.y + p.z)), 0.1));
    res = comb(res, vec4(sdCapsule((p - vec3(0.0, 0.9, 0.0)), vec3(0.0 + waveOff, 0.0, 0.0), vec3(0.0, -0.9, 0.0), 0.1), 0.8, 0.7, 0.4));

    return res;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.33); // body
    d = smin(d, sdSphere((p - vec3(0.0, 0.7, -0.7)), 0.20), 0.1); // head
    d = smin(d, sdCapsule((p - vec3(0.1, 0.0, -0.7)), vec3(0.0, 0.05, 0.0), vec3(-0.01, -0.04, 0.0), 0.05), 0.01); // leg left (for monster)
    d = smin(d, sdCapsule((p - vec3(-0.1, 0.0, -0.7)), vec3(0.0, 0.05, 0.0), vec3(0.01, -0.04, 0.0), 0.05), 0.01); // leg right
    d = smin(d, sdCapsule((p - vec3(0.35, 0.35, -0.7)), vec3(-0.03, 0.05, 0.0), vec3(0.09, -0.04, 0.15), 0.05), 0.01); // hand left (for monster)
    d = smin(d, sdCapsule((p - vec3(-0.35, 0.35, -0.7)), vec3(0.03, 0.05, 0.0), vec3(-0.09, -0.04 + 0.1 * lazysin(10.0 * iTime), 0.05 + 0.1 * lazycos(10.0 * iTime)), 0.05), 0.01); // hand right

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}


const float PI = 3.14159265;

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);

    vec3 off = vec3(0.0, 0.6, -0.50);
    float angle = lazysin((iTime + 9.0) * 4.0, 1) * PI / 8.0;
    vec3 eyeDir = vec3(sin(angle), 0.0, cos(angle));
    vec3 off2 = off + 0.02 * eyeDir;

    res = comb(res, vec4(sdSphere((p - off), 0.22), 1.0, 1.0, 1.0));
    res = comb(res, vec4(sdSphere((p - off2), 0.202), (dot(p - off2, eyeDir) < 0.196 ? vec3(0.1, 1.0, 1.0) : vec3(0.1, 0.1, 0.1))));
    // res = comb(res, vec4(sdSphere((p - off3), 0.1828), ));

    return res;
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
        float a = 0.5 + 2.0 * sin(abs(p.x) + abs(p.z));
        res = vec4(dist, a * vec3(1.0, 1.0, 0.5) + (1.0 - a) * vec3(0.5, 0.9, 0.5));
    }
    res = comb(res, sdTree(p - vec3(1.0, 0.0, -1.6)));
    res = comb(res, sdTree(p - vec3(-0.9, 0.0, -1.4)));
    res = comb(res, sdTree(p - vec3(3.3, 0.0, -5.4)));
    res = comb(res, sdTree(p - vec3(-4.0, 0.0, -5.4)));

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                      sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                      sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
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
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
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


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));


    vec4 res = raycast(ray_origin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}
