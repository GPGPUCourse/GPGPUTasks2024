// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p) {
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle) {
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

float smin(float a, float b, float k) {
    k *= 1.0;
    float r = exp2(-a/k) + exp2(-b/k);
    return -k*log2(r);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

#define MAT_BODY 1
#define MAT_EYE_OUTER 2
#define MAT_EYE_EDGE 3
#define MAT_EYE_INNER 4
#define MAT_GROUND 5

struct Mat {
    vec3 albedo;
    float glossiness;
    float roughness;
    float reflectiveness;
};

const Mat mats[] = Mat[](
Mat(vec3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0),
Mat(vec3(0.0, 1.0, 0.0), 0.2, 0.3, 0.0), // MAT_BODY
Mat(vec3(1.0, 1.0, 1.0), 0.5, 0.5, 0.0), // MAT_EYE_OUTER
Mat(vec3(0.0, 0.8, 1.0), 0.5, 0.5, 0.0), // MAT_EYE_EDGE
Mat(vec3(0.0, 0.0, 0.0), 0.9, 0.1, 0.5), // MAT_EYE_INNER
Mat(vec3(0.3, 0.0, 0.0), 0.2, 0.3, 0.6)  // MAT_GROUND
);

float sdBody(vec3 p, inout int mat) {
    float d = 1e10;

    mat = MAT_BODY;

    d = smin(
        sdSphere((p - vec3(0.0, 0.45, -0.7)), 0.35),
        sdSphere((p - vec3(0.0, 0.7, -0.6)), 0.25),
        0.04
    );

    float anim = lazycos(iTime * 8.0);
    d = min(d, sdCapsule(p, vec3(0.3, 0.45, -0.6), vec3(0.35, 0.35, -0.58), 0.05));
    d = min(d, sdCapsule(p, vec3(-0.3, 0.45, -0.6), vec3(-0.35, 0.45 - anim * 0.1, -0.58), 0.05));
    d = min(d, sdCapsule(p, vec3(0.1, 0.2, -0.7), vec3(0.1, 0.05, -0.7), 0.05));
    d = min(d, sdCapsule(p, vec3(-0.1, 0.2, -0.7), vec3(-0.1, 0.05, -0.7), 0.05));

    return d;
}

float sdEye(vec3 p, inout int mat) {
    float d = 1e10;
    p -= vec3(0.0, 0.67, -0.46);
    d = sdSphere(p, 0.2);
    vec3 looking_at = normalize(vec3(sin(iTime * 0.5) * 0.2, sin(iTime * 1.1) * 0.2, 1.0));
    float angle_cos = dot(normalize(looking_at), normalize(p));

    if (angle_cos > 0.95) {
        mat = MAT_EYE_INNER;
    } else if (angle_cos > 0.85) {
        mat = MAT_EYE_EDGE;
    } else {
        mat = MAT_EYE_OUTER;
    }
    return d;
}


#define UPDATE0(sdf, out_mat) { float out_dist = sdf(p); if (out_dist < dist) { mat = out_mat; dist = out_dist; } }
#define UPDATE1(sdf) { int out_mat; float out_dist = sdf(p, out_mat); if (out_dist < dist) { mat = out_mat; dist = out_dist; } }
#define UPDATE2(sdf) { int out_mat; float out_dist = dist; sdf(p, out_dist, out_mat); if (out_dist < dist) { mat = out_mat; dist = out_dist; } }

void sdMonster(vec3 p, inout float dist, inout int mat) {
    UPDATE1(sdBody);
    UPDATE1(sdEye);
}

float sdRoundedCylinder( vec3 p, float ra, float rb, float h ) {
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b + r;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

void sdTotal(vec3 p, inout float dist, inout int mat) {
    UPDATE0(sdPlane, MAT_GROUND);
    UPDATE2(sdMonster);
}

float sdTotalDist(vec3 p) {
    int _;
    float dist = 1e10;
    sdTotal(p, dist, _);
    return dist;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) {
    const float eps = 0.0001;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotalDist(p+h.xyy) - sdTotalDist(p-h.xyy),
                      sdTotalDist(p+h.yxy) - sdTotalDist(p-h.yxy),
                      sdTotalDist(p+h.yyx) - sdTotalDist(p-h.yyx) ) );
}


float raycast(vec3 ray_origin, vec3 ray_direction, out int mat)
{

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        float dist = 1e10;
        sdTotal(ray_origin + t*ray_direction, dist, mat);
        t += dist;
        if (dist < EPS) {
            return t;
        }
    }

    mat = 0;
    return 1e10;
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float power) {
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), power);
}


float castShadow(vec3 p, vec3 light_source)
{

    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);
    int _;


    if (raycast(light_source, normalize(light_dir), _) + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}

void materialized_cast(vec3 ray_origin, vec3 ray_direction, out vec3 col, out Mat mat, out vec3 point, out vec3 normal) {
    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);
    int mat_index;

    float dist = raycast(ray_origin, ray_direction, mat_index);

    mat = mats[mat_index];

    if (mat_index == 0) {
        float theta = iTime * 0.1;
        col = texture(iChannel0, ray_direction * mat3(
        cos(theta),  0.0,-sin(theta),
        0.0, 1.0, 0.0,
        sin(theta), 0.0, cos(theta)
        )).rgb;
    } else {
        col = mat.albedo;
        point = ray_origin + dist*ray_direction;
        normal = calcNormal(point);

        float shad = shading(point, light_source, normal);
        shad = min(shad, castShadow(point, light_source));
        col *= shad;

        float spec = mat.glossiness * specular(point, light_source, normal, ray_origin, 1.0 / (mat.roughness * mat.roughness) - 1.0);
        col += vec3(1.0, 1.0, 1.0) * spec;
    }
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));

    Mat mat, reflected_mat;
    vec3 color, hit_point, normal, hit_point2, reflected_color;

    materialized_cast(ray_origin, ray_direction, color, mat, hit_point, normal);
    if (mat.reflectiveness > 0.0) {
        ray_direction = reflect(ray_direction, normal);
        materialized_cast(hit_point + normal * 1e-3, ray_direction, reflected_color, reflected_mat, hit_point2, normal);
        color += reflected_color * mat.reflectiveness;
    }

    // Output to screen
    fragColor = vec4(color, 1.0);
}
