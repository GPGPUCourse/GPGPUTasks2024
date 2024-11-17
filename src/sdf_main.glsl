
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

// took from https://iquilezles.org/articles/distfunctions/
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

float smin( float a, float b, float k )
{
    k *= 6.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*h*k*(1.0/6.0);
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

vec3 calcShift( in vec3 p, float x_angle, float y_angle, float z_angle)
{
    float pi = 3.;
    x_angle = x_angle / 180. * pi;
    y_angle = y_angle / 180. * pi;
    z_angle = z_angle / 180. * pi;
    mat3 ox = mat3(1, 0, 0, 0, cos(x_angle), -sin(x_angle), 0, sin(x_angle), cos(x_angle));
    mat3 oy = mat3(cos(y_angle), 0, sin(y_angle), 0, 1, 0, -sin(y_angle), 0, cos(y_angle));
    mat3 oz = mat3(cos(z_angle), -sin(z_angle), 0, sin(z_angle), cos(z_angle), 0, 0, 0, 1);
    return oz * oy * ox * vec3(p.x, p.y, p.z);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    float body = sdRoundCone(p - vec3(0.0, 0.35, -0.7), 0.25, 0.2, 0.2 );
    float left_leg = sdRoundCone(p - vec3(0.1, 0.1, -0.7), 0.06, 0.05, 0.08 );
    float right_leg = sdRoundCone(p - vec3(-0.1, 0.1, -0.7), 0.06, 0.05, 0.08 );
    float left_hand = sdRoundCone(calcShift(p  - vec3(0.23, 0.35, -0.7), 0., 0., -150.), 0.04, 0.035, 0.08 );
    float right_hand = sdRoundCone(calcShift(p  - vec3(-0.23, 0.35, -0.7), 0., 0., 50. + 50. * (1. + lazycos(iTime * 8.))), 0.04, 0.035, 0.08 );
    d = smin(body, left_leg, 0.005);
    d = smin(d, right_leg, 0.005);
    d = smin(d, right_hand, 0.005);
    d = smin(d, left_hand, 0.005);
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 res;
    float levels;
    vec3 white_color = vec3(1);
    vec3 blue_color = vec3(0, 1, 1);
    vec3 black_color = vec3(0);
    float circle_1 = sdSphere(p - vec3(0.0, 0.59, -0.55), 0.12);
    float circle_2 = sdSphere(p - vec3(0.0, 0.59, -0.504), 0.09);
    float circle_3 = sdSphere(p - vec3(0.0, 0.59, -0.48), 0.07);
    levels = circle_1;
    res = white_color;
    if (circle_2 < levels) {
        levels = circle_2;
        res = blue_color;
    }
    if (circle_3 < levels) {
        levels = circle_3;
        res = black_color;
    }
    return vec4(levels, res);
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
