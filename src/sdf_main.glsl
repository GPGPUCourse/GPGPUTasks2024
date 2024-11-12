
float smin(float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

float sdTorus( vec3 p, vec2 t )
{
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}


float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

vec4 cmin(vec4 a, vec4 b)
{
    return a.x < b.x ? a : b;
}

float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdPlane(vec3 p)
{
    return p.y;
}


float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float PI = 3.1415926;

float lazycos(float angle, int times, int skip)
{
    int nsleep = times;

    int iperiod = int(angle / (2.0 * PI)) % nsleep;
    if (iperiod <  skip) {
        return cos(angle);
    }

    return 1.0;
}

float lazysin(float angle, int times, int skip)
{
    int nsleep = times;

    int iperiod = int(angle / (2.0 * PI)) % nsleep;
    if (iperiod < skip) {
        return sin(angle);
    }

    return 0.0;
}

float lazysininv(float angle, int times, int skip)
{
    int nsleep = times;

    int iperiod = int(angle / (2.0 * PI)) % nsleep;
    if (iperiod >= skip) {
        return sin(angle);
    }

    return 0.0;
}

vec4 sdBody(vec3 p)
{
    float d = 1e10;


    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.3);
    d = smin(d, sdSphere((p - vec3(0.0, 0.6, -0.7)), 0.2), 0.1);

    vec3 a = vec3(-0.3, 0.4, -0.7);
    vec3 b = vec3(-0.4, 0.4, -0.7);
    float angle = (PI * iTime * 2.0);

    d = smin(d, sdCapsule(p, vec3(0.3, 0.4, -0.7), vec3(0.4, 0.4, -0.7) + vec3(0.0, 0.1 * -(lazycos(angle, 3, 2)), 0.1 * abs(lazysin(angle, 3, 2))), 0.05), 0.001);
    d = smin(d, sdCapsule(p, a, b + vec3(0.0, 0.1 * -(lazycos(angle, 3, 2)), 0.1 * abs(lazysin(angle, 3, 2))), 0.05), 0.001);

    d = smin(d, sdCapsule(p, vec3(-0.1, 0, -0.7), vec3(-0.1, 0.1, -0.7), 0.05), 0.001);
    d = smin(d, sdCapsule(p, vec3(0.1, 0, -0.7), vec3(0.1, 0.1, -0.7), 0.05), 0.001);


    return vec4(d,  vec3(0.6, 0.6, 0.6));
}

vec4 sdEye(vec3 p)
{

    p -= vec3(0, 0.5, -0.5);

    float angle = PI / 8.0 * lazysininv((PI * iTime * 2.0), 3, 2);
    p = vec3(p.x * cos(angle) - p.z * sin(angle), p.y, p.z * cos(angle) + p.x * sin(angle));

    vec4 res = vec4(sdSphere(p, 0.2), 1.0, 1.0, 1.0);
    res = cmin(res, vec4(sdSphere(p - vec3(0.0, 0, 0.15), 0.07), 0.0, 0.0, 0.0));


    return res;
}




vec4 sdMonster(vec3 p)
{

    p -= vec3(0.0, 0.08, 0.0);
    p -= vec3(0, 0.1 * abs(lazysin(2.0 * PI * iTime, 3, 2)), 0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    res = cmin(res, vec4(sdTorus(p - vec3(0, 0.85 + 0.025 * abs(lazysin(2.0 * PI * iTime, 3, 2)), -0.7), vec2(0.25, 0.02)), 1.0, 215.0/255.0, 0.0));


    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(50.0 / 255.0, 205.0 / 255.0, 50.0 / 255.0) );
    }

    return res;
}


vec3 calcNormal( in vec3 p )
{
    const float eps = 0.0001;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                      sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                      sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{

    float EPS = 1e-3;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, 2.0 * vec3(135.0 / 255.0, 206.0 / 255.0, 250.0 / 255.0) * (0.8 + 0.2 * sin(iTime / 5.0)));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);

}

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

    vec3 light_source = vec3(5.0 + 2.5*sin(iTime) * 0.0, 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;


    fragColor = vec4(col, 1.0);
}
