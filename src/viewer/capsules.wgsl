// Capsule impostors (cylinders with rounded ends): one camera-facing quad per capsule;
// the fragment shader ray-casts the exact capsule and writes its depth. Rounded ends
// make chains of sticks join without gaps.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct Instance {
    @location(0) start: vec3<f32>,
    @location(1) radius: f32,
    @location(2) end: vec3<f32>,
    @location(3) start_color: vec4<f32>,
    @location(4) end_color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) start: vec3<f32>,
    @location(2) end: vec3<f32>,
    @location(3) radius: f32,
    @location(4) start_color: vec4<f32>,
    @location(5) end_color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) corner: u32, c: Instance) -> VertexOut {
    let uv = vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) * 2.0 - 1.0;
    let mid = 0.5 * (c.start + c.end);
    let half_axis = 0.5 * (c.end - c.start);
    let to_eye = normalize(camera.eye.xyz - mid);
    // Quad spans the axis (plus the caps) and the radius across it, facing the eye.
    var along = half_axis;
    if (length(along) < 1e-6) {
        along = camera.right.xyz * 1e-3;
    }
    let side = normalize(cross(along, to_eye));
    let axis_dir = normalize(along);
    let length_half = length(half_axis) + c.radius;
    let world = mid + to_eye * c.radius
        + axis_dir * uv.x * length_half * 1.2
        + side * uv.y * c.radius * 1.5;
    var out: VertexOut;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.start = c.start;
    out.end = c.end;
    out.radius = c.radius;
    out.start_color = c.start_color;
    out.end_color = c.end_color;
    return out;
}

// Distance along the ray to the capsule, or −1 (Quílez's capsule intersection).
fn hit_capsule(ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, r: f32) -> f32 {
    let ba = pb - pa;
    let oa = ro - pa;
    let baba = dot(ba, ba);
    let bard = dot(ba, rd);
    let baoa = dot(ba, oa);
    let rdoa = dot(rd, oa);
    let oaoa = dot(oa, oa);
    let a = baba - bard * bard;
    var b = baba * rdoa - baoa * bard;
    var c = baba * oaoa - baoa * baoa - r * r * baba;
    var h = b * b - a * c;
    if (h >= 0.0) {
        let t = (-b - sqrt(h)) / a;
        let y = baoa + t * bard;
        if (y > 0.0 && y < baba) {
            return t;
        }
        var oc = oa;
        if (y > 0.0) {
            oc = ro - pb;
        }
        b = dot(rd, oc);
        c = dot(oc, oc) - r * r;
        h = b * b - c;
        if (h > 0.0) {
            return -b - sqrt(h);
        }
    }
    return -1.0;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(v: VertexOut) -> FragmentOut {
    let ro = camera.eye.xyz;
    let rd = normalize(v.world - ro);
    let t = hit_capsule(ro, rd, v.start, v.end, v.radius);
    if (t < 0.0) {
        discard;
    }
    let hit = ro + rd * t;
    let ba = v.end - v.start;
    let along = clamp(dot(hit - v.start, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
    let normal = normalize(hit - (v.start + along * ba));
    var color = v.start_color;
    if (along > 0.5) {
        color = v.end_color;
    }

    let light = normalize(ro - hit);
    let diffuse = max(dot(normal, light), 0.0);
    let rim = pow(1.0 - diffuse, 3.0) * 0.15;
    let shade = 0.35 + 0.65 * diffuse;

    let clip = camera.view_proj * vec4<f32>(hit, 1.0);
    var out: FragmentOut;
    out.color = vec4<f32>(color.rgb * shade + rim, color.a);
    out.depth = clip.z / clip.w;
    return out;
}
