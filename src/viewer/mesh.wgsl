// Triangle meshes (molecular surface, arrow heads) with per-vertex colour and
// normal; from purrette's, without contour lines.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(v: Vertex) -> VertexOut {
    var out: VertexOut;
    out.clip = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.world = v.position;
    out.normal = v.normal;
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(v: VertexOut) -> @location(0) vec4<f32> {
    // Same headlight as the impostors, so all layers read alike.
    let light = normalize(camera.eye.xyz - v.world);
    let diffuse = abs(dot(normalize(v.normal), light));
    let shade = 0.35 + 0.65 * diffuse;
    return vec4<f32>(v.color.rgb * shade, v.color.a);
}
