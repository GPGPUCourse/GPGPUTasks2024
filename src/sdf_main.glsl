
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
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
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

// quadratic polynomial
float smin( float a, float b, float k )
{
    k *= 4.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*k*(1.0/4.0);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float K_SMIN = 0.025;
    float K_SMIN_HANDS = 0.01;
    float HAND_MOVEMENT_SPEEDUP = 7.0;
    float LEG_K = 0.03;

    float d = 1e10;

    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.3);
    d = smin(d, sdSphere((p - vec3(0.0, 0.6, -0.7)), 0.20), K_SMIN);

#define LEG_P 0.125, 0.01, -0.7
#define LEG_A 0.0, 0.7, 0.0
#define LEG_B 0.03, -0.05, 0.01
    d = smin(d, sdCapsule((p - vec3(LEG_P)), vec3(LEG_A), vec3(-LEG_B), LEG_K), K_SMIN);
    d = smin(d, sdCapsule((p - vec3(-LEG_P)), vec3(LEG_A), vec3(LEG_B), LEG_K), K_SMIN);
#undef LEG_B
#undef LEG_A
#undef LEG_P

#define HAND_P 0.27, 0.36, -0.6
#define HAND_A 0.09, 0.1, 0.0
    d = smin(d, sdCapsule((p - vec3(HAND_P)), vec3(-HAND_A), vec3(0.1, -0.05, 0.15), 0.04), K_SMIN_HANDS);
    d = smin(d, sdCapsule((p - vec3(-HAND_P)), vec3(HAND_A), vec3(-0.1, -0.05 + 0.07 * sin(HAND_MOVEMENT_SPEEDUP * iTime), 0.05 + 0.07 * lazycos(HAND_MOVEMENT_SPEEDUP * iTime)), 0.04), K_SMIN_HANDS);
#undef HAND_A
#undef HAND_P

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    float radiusEye = 0.15;
    float radiusRim = 0.08;
    float radiusPupil = 0.05;

    float dEye = sdSphere(p + vec3(0.0, -0.5, 0.4), radiusEye);
    float dRim = sdSphere(p + vec3(0.0, -0.5, 0.3), radiusRim);
    float dPupil = sdSphere(p + vec3(0.0, -0.5, 0.25), radiusPupil);

    vec4 res = vec4(dEye, vec3(1.0, 1.0, 1.0));
    vec4 rim = vec4(dRim, vec3(0.1, 0.4, 1.0));
    vec4 pupil = vec4(dPupil, vec3(0.0, 0.0, 0.0));

    if (rim.x < res.x) {
        res = rim;
    }
    if (pupil.x < res.x) {
        res = pupil;
    }
    
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
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }
    
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
