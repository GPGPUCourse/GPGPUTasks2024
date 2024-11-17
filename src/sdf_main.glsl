
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
    int nsleep = 4;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 2) {
        return cos(angle);
    }
    
    return 1.0;
}

float lazycos2(float angle)
{
    int nsleep = 4;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod >= 2) {
        return cos(angle);
    }
    
    return 1.0;
}

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

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

// circular
float smin( float a, float b, float k )
{
    k *= 1.0/(1.0-sqrt(0.5));
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - k*0.5*(1.0+h-sqrt(1.0-h*(h-2.0)));
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;
    
    vec3 pa_global = vec3(0., 0.35, 0.);
    vec3 hands_up = vec3(0., 0.1, 0.);
    vec3 legs_down = vec3(0., -0.1, 0.);
    
    float smoothing  = 0.03 + cos(iTime*5.)*0.01;

    // body
    d = sdRoundCone(p, pa_global, (vec3(0., 0.6, 0.)), 0.3, 0.2);

    // right hand
    vec3 pbrh = vec3(0.4, 0.45, 0.);
    vec3 time_moverh = lazycos(iTime*10.)*vec3(0., 0.15, 0.);
    d = smin(sdRoundCone(p, pa_global + hands_up, pbrh - time_moverh, 0.1, 0.07), d, smoothing);

    // left hand
    vec3 pblh = vec3(-0.4, 0.45, 0.);
    vec3 time_movelh = lazycos2(iTime*10.)*vec3(0., 0.15, 0.);
    d = smin(sdRoundCone(p, pa_global + hands_up, pblh - time_movelh, 0.1, 0.07), d, smoothing);

    vec3 leg_lr = vec3(0.1, 0., 0.);
    vec3 leg_move = vec3(cos(iTime*10.)*0.05, 0., 0.);
    // left leg
    vec3 pbll = vec3(-0.15, 0., 0.);
    d = smin(sdRoundCone(p, pa_global + legs_down - leg_lr, pbll + leg_move, 0.1, 0.07), d, smoothing);
    // right leg
    vec3 pbrl = vec3(0.15, 0., 0.);
    d = smin(sdRoundCone(p, pa_global + legs_down + leg_lr, pbrl + leg_move, 0.1, 0.07), d, smoothing);


    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 p_1 = vec3(0., 0.55, 0.15);
    vec3 p_2 = vec3(0., 0.55, 0.25);
    vec3 p_3 = vec3(0., 0.55, 0.3);
    
    float d1 = sdSphere(p - p_1, 0.15);
    float d2 = sdSphere(p - p_2, 0.07);
    float d3 = sdSphere(p - p_3, 0.03);
    
    if (d1 < d2 && d1 < d3)
    {
        return vec4(d1, vec3(1., 1., 1.));
    }
    else if (d2 < d1 && d2 < d3)
    {
    return vec4(d2, vec3(0., 0., 1.));
    }
    return vec4(d2, vec3(0., 0., 0.));
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
    

    vec3 ray_origin = vec3(cos(iTime*2.5)*0.3, 0.5, 1.0);
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
