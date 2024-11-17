
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

//https://iquilezles.org/articles/distfunctions/
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

//https://iquilezles.org/articles/smin/
float smin( float a, float b, float k )
{
    k *= 6.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*h*k*(1.0/6.0);
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

vec3 rotateZ(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(
        c * p.x - s * p.y,
        s * p.x + c * p.y,
        p.z
    );
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    float angle = -3.5 * lazycos(2.0 * iTime) + 0.4;
    float body = sdRoundCone(p - vec3(0.0, 0.35, -0.7), 0.3, 0.25, 0.25);
    float left_leg = sdRoundCone(p - vec3(0.15, -0.1, -0.7), 0.06, 0.06, 0.4);
    float right_leg = sdRoundCone(p - vec3(-0.15, -0.1, -0.7), 0.05, 0.05, 0.4);
    vec3 rotated_left_arm = rotateZ(rotateZ(p - vec3(0.43, 0.35, -0.7), radians(-40.0)), 0.0);
    float left_arm = sdRoundCone(rotated_left_arm, 0.05, 0.05, 0.25);
    vec3 rotated_right_arm = rotateZ(rotateZ(p - vec3(-0.27, 0.5, -0.7), radians(40.0)), angle);
    float right_arm = sdRoundCone(rotated_right_arm, 0.05, 0.05, 0.25);
    
    d = smin(body, left_leg, 0.005);
    d = smin(d, right_leg, 0.005);
    d = smin(d, right_arm, 0.005);
    d = smin(d, left_arm, 0.005);
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}


vec4 sdEye(vec3 p)
{
    vec3 eyeCenter = vec3(0.0, 0.5, -0.5);
    vec3 offset = p - eyeCenter;

    float whiteRadius = 0.12;
    float blueRadius = 0.07;
    float blackRadius = 0.04;

    float whiteLayer = sdSphere(offset, whiteRadius);
    vec3 whiteColor = vec3(1.0, 1.0, 1.0);

    float blueLayer = sdSphere(p - vec3(0.0, 0.51, -0.41), blueRadius);
    vec3 blueColor = vec3(0.0, 1.0, 1.0);

    float blackLayer = sdSphere(p - vec3(0.0, 0.51, -0.3), blackRadius);
    vec3 blackColor = vec3(0.0, 0.0, 0.0);
    
    float layer = whiteLayer;
    vec3 color = whiteColor;
    
    if (blueLayer < layer) {
        layer = blueLayer;
        color = blueColor;
    }
    if (blackLayer < layer) {
        layer = blackLayer;
        color = blackColor;
    }
    return vec4(layer, color);
}


vec4 sdMonster(vec3 p)
{
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 res = sdBody(p);
    
    vec4 centralEye = sdEye(p);
    if (centralEye.x < res.x) {
        res = centralEye;
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
