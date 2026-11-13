// Sphere impostors: one camera-facing quad per sphere; the fragment shader ray-casts
// the exact sphere and writes its depth, so spheres intersect correctly.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct Instance {
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) corner: u32, s: Instance) -> VertexOut {
    let uv = vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) * 2.0 - 1.0;
    // Slightly oversized and pulled towards the eye so perspective never clips the silhouette.
    let to_eye = normalize(camera.eye.xyz - s.center);
    let base = s.center + to_eye * s.radius;
    let world = base + (camera.right.xyz * uv.x + camera.up.xyz * uv.y) * s.radius * 1.5;
    var out: VertexOut;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.center = s.center;
    out.radius = s.radius;
    out.color = s.color;
    return out;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(v: VertexOut) -> FragmentOut {
    let dir = normalize(v.world - camera.eye.xyz);
    let oc = camera.eye.xyz - v.center;
    let b = dot(oc, dir);
    let disc = b * b - (dot(oc, oc) - v.radius * v.radius);
    if (disc < 0.0) {
        discard;
    }
    let hit = camera.eye.xyz + dir * (-b - sqrt(disc));
    let normal = normalize(hit - v.center);

    // Headlight plus a little ambient: shape reads from any angle without a light setup.
    let light = normalize(camera.eye.xyz - hit);
    let diffuse = max(dot(normal, light), 0.0);
    let rim = pow(1.0 - diffuse, 3.0) * 0.15;
    let shade = 0.35 + 0.65 * diffuse;

    let clip = camera.view_proj * vec4<f32>(hit, 1.0);
    var out: FragmentOut;
    out.color = vec4<f32>(v.color.rgb * shade + rim, v.color.a);
    out.depth = clip.z / clip.w;
    return out;
}
