
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
    int nsleep = 10;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }
    
    return 1.0;
}

float smin( float a, float b, float k )
{
    k *= 1.0;
    float r = exp2(-a/k) + exp2(-b/k);
    return -k*log2(r);
}

float opUnion( float d1, float d2 )
{
    return smin(d1,d2,0.015);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    d = opUnion(sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35), sdSphere((p - vec3(0.0, 0.60, -0.7)), 0.30));
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    
    float d1 = sdSphere((p), 0.2);
    float d2 = sdSphere((p - vec3(0.0, 0.0, 0.11)), 0.1);
    float d3 = sdSphere((p - vec3(0.0, 0.0, 0.088)), 0.12);
    if (d1 < d2 && d1 < d3) {
        return vec4(d1, vec3(1.0, 1.0, 1.0));
    } else if (d2 < d1 && d2 < d3) {
        return vec4(d2, vec3(0.0, 0.0, 0.0));
    } else {
        return vec4(d3, vec3(0.0, 0.0, 1.0));
    }
    
}

float sdVerticalCapsule( vec3 p, float h, float r )
{
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
}

vec4 sdLeg(vec3 p)
{
    return vec4(sdVerticalCapsule(p, 0.2, 0.05), 0.0, 1.0, 0.0);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

#define M_PI 3.1415926535897932384626433832795

vec4 sdRightArm(vec3 p)
{
    int n = int(iTime / (M_PI / 2.0));
    bool even = n % 2 == 0;
    float phi = even ? (2.0 * M_PI / 3.0) + iTime - (M_PI / 2.0) * float(n) : (7.0 * M_PI / 6.0) - (iTime - (M_PI / 2.0) * float(n));
    float l = 0.3;
    return vec4(sdCapsule(p, vec3(-l*abs(cos(phi)), l*sin(phi), 0.0), vec3(0.0, 0.0, 0.0), 0.05), 0.0, 1.0, 0.0);
}

vec4 sdLeftArm(vec3 p)
{
    int n = int(iTime / (M_PI / 2.0));
    bool even = n % 2 == 0;
    float phi = even ? (2.0 * M_PI / 3.0) + iTime - (M_PI / 2.0) * float(n) : (7.0 * M_PI / 6.0) - (iTime - (M_PI / 2.0) * float(n));
    float l = 0.3;
    return vec4(sdCapsule(p, vec3(l*abs(cos(phi)), l*sin(phi), 0.0), vec3(0.0, 0.0, 0.0), 0.05), 0.0, 1.0, 0.0);
}

vec4 choose(vec4 lhs, vec4 rhs) {
    return lhs.x < rhs.x ? lhs : rhs;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 body = sdBody(p);
    vec4 eye = sdEye(p - vec3(0.0, 0.62, -0.57));
    vec4 legRight = sdLeg(p - vec3(-0.07, 0.0, -0.7));
    vec4 legLeft = sdLeg(p - vec3(0.07, 0.0, -0.7));
    vec4 armRight = sdRightArm(p - vec3(-0.25, 0.44, -0.7));
    vec4 armLeft = sdLeftArm(p - vec3(0.25, 0.44, -0.7));
    
    return choose(body, choose(eye, choose(legRight, choose(legLeft, choose(armRight, armLeft)))));
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
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.01; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p).x,
                           sdTotal(p+h.yxy).x - sdTotal(p).x,
                           sdTotal(p+h.yyx).x - sdTotal(p).x ) );
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
    

    vec3 ray_origin = vec3(0.0, 0.5, 1);
    vec3 px = vec3(uv.x - 0.5*wh.x, uv.y, 0.0);
    vec3 ray_direction = normalize(px - ray_origin);


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
