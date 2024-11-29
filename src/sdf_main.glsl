vec3 GREEN = vec3(0.0, 1.0, 0.0);

float smin( float a, float b, float k )
{
    k *= 16.0/3.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*h*(4.0-h)*k*(1.0/16.0);
}

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdEllipsoid(vec3 p, vec3 r)
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

mat3 getRotationMatrixZ(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
        c, -s, 0.0,
        s,  c, 0.0,
        0.0, 0.0, 1.0
    );
}

float sdRoundCone( vec3 p, float r1, float r2, float h )
{
  // sampling independent computations (only depend on shape)
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);

  // sampling dependant computations
  vec2 q = vec2( length(p.xz), p.y );
  float k = dot(q,vec2(-b,a));
  if( k<0.0 ) return length(q) - r1;
  if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
  return dot(q, vec2(a,b) ) - r1;
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

vec4 sdLegs(vec3 p)
{

    float d1 = sdRoundCone((p - vec3(0.3, 0.05, -0.5)), 0.073, 0.05, 0.1);
    float d2 = sdRoundCone((p - vec3(0.7, 0.05, -0.5)), 0.073, 0.05, 0.1);


    if (d1 < d2) {
        return vec4(d1, GREEN);
    }
    return vec4(d2, GREEN);
}

vec4 sdHands(vec3 p)
{

    float rotationAngle = -0.9 * lazycos(2.0 * iTime) + 4.0;

    mat3 rotationMatrixRotate = getRotationMatrixZ(rotationAngle);
    vec3 p1 = p - vec3(0.1, 0.7, -0.6);
    vec3 p2 = p - vec3(1.1, 0.45, -0.65);
    mat3 rotationMatrix1 = getRotationMatrixZ(0.6);
    mat3 rotationMatrix2 = getRotationMatrixZ(-0.6);

    float d1 = sdRoundCone(p1 * rotationMatrix1 * rotationMatrixRotate, 0.05, 0.05, 0.3);
    float d2 = sdRoundCone(p2 * rotationMatrix2, 0.05, 0.05, 0.3);


    if (d1 < d2) {
        return vec4(d1, GREEN);
    }
    return vec4(d2, GREEN);
}

vec4 sdLimbs(vec3 p)
{

    vec4 d1 = sdLegs(p);
    vec4 d2 = sdHands(p);


    if (d1.x < d2.x) {
        return d1;
    }
    return d2;
}

vec4 sdBodyMain(vec3 p) {
    float e = sdEllipsoid((p - vec3(0, 0.7, -0.7)), vec3(0.31, 0.6, 0.11));
    float s = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);

    float d = min(e, s) + smin(e, s, 0.3);
    return vec4(d, GREEN);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{

    vec4 body = sdBodyMain(p - vec3(0.5, 0.1, 0));
    vec4 limbs = sdLimbs(p);

    if (body.x < limbs.x) {
        return body;
    }
    return vec4(limbs);
}

vec4 sdEye(vec3 p)
{
    float blinkRate = abs(lazycos(iTime + 2.0));

    float d1 = sdEllipsoid((p - vec3(0.45, 0.41, -0.2)), vec3(0.14, 0.14 * blinkRate, 0.1));
    vec3 blue = vec3(0.25, 0.99, 0.81);


    float d2 = sdEllipsoid((p - vec3(0.45, 0.43, -0.4)), vec3(0.28, 0.25 * blinkRate, 0.25));
    vec3 white = vec3(1.0, 1.0, 1.0);

    float d3 = sdEllipsoid((p - vec3(0.45, 0.39, -0.1)), vec3(0.07, 0.07 * blinkRate, 0.01));
    vec3 black = vec3(0.0, 0.0, 0.0);

    if (d1 < d2 && d1 < d3) {
        return vec4(d1, blue);
    } else if (d2 < d1 && d2 < d3) {
        return vec4(d2, white);
    } else {
        return vec4(d3, black);
    }
}



vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(-0.5, 0.0, -0.5);

    vec4 res = sdBody(p);

    p -= vec3(0.04, 0.5, 0.0);

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
    
    
    if (raycast(light_source, normalize(light_dir)).x + 1.0 < target_dist) {
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
