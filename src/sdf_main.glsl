
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

float dot2(in vec3 v ) { return dot(v,v); }
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    // sampling independent computations (only depend on shape)
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;
    
    // sampling dependant computations
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot2( pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

float smin( float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // Main body
    float d1 = sdSphere((p - vec3(0.0, 0.25, -0.7)), 0.27);
    float d2 = sdRoundCone(
        p, 
        vec3(0.0, 0.25, -0.5),
        vec3(0.0, 0.5, -0.5), 
        0.1,
        0.14
    );
    d = smin(d1, d2, 0.1);
    
    // Legs dancing
    float end_z1 = -0.6 - 0.01 * lazycos(iTime); 
    float end_x1 = -0.1 + 0.1 * lazycos(iTime); 
    float end_z2 = -0.6 - 0.01 * lazycos(iTime); 
    float end_x2 = 0.1 + 0.1 * lazycos(iTime); 
    float leg_d1 = sdCapsule(p, vec3(end_x1, -0.05, end_z1), vec3(-0.1, 0.05, -0.6), 0.05);
    float leg_d2 = sdCapsule(p, vec3(end_x2, -0.05, end_z2), vec3(0.1, 0.05, -0.6), 0.05);
    
    d = min(d, leg_d1);
    d = min(d, leg_d2);
    
    // Left arm (stationary)
    float left_arm_d = sdCapsule(p, vec3(0.32, 0.17, -0.6), vec3(0.25, 0.3, -0.6), 0.05);
    d = min(d, left_arm_d);
    
    // Right arm (waving)
    float end_y = 0.25 - 0.07 * lazycos(iTime); 
    float right_arm_d = sdCapsule(p, vec3(-0.32, end_y, -0.6), vec3(-0.25, 0.3, -0.6), 0.05);
    
    d = min(d, right_arm_d);
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 closest(vec4 v1, vec4 v2) {
    if (v1.x < v2.x) {
        return v1;
    }
    return v2;
}

vec4 sdEye(vec3 p)
{
    
    float white_d = sdSphere((p - vec3(0.0, 0.48, -0.4)), 0.14);
    vec4 white = vec4(white_d, vec3(1.0, 1.0, 1.0));
    
    float blue_d = sdSphere((p - vec3(0.0, 0.48, -0.31)), 0.085);
    vec4 blue = vec4(blue_d, vec3(0.0, 1.0, 1.0));
    
    float black_d = sdSphere((p - vec3(0.0, 0.48, -0.27)), 0.055);
    vec4 black = vec4(black_d, vec3(0.0, 0.0, 0.0));
    
    vec4 res = closest(closest(white, blue), black);
    
    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.10, 0.0);
    
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