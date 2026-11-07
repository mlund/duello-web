//! wgpu side of the viewer, from purrette's: pipelines live in egui-wgpu's
//! callback resources, and each frame's instances are uploaded in `prepare`.

use eframe::egui_wgpu::{self, wgpu};

use super::{Camera, Cylinder, Scene, Sphere};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

impl From<&Camera> for CameraUniform {
    fn from(c: &Camera) -> Self {
        let (right, up) = c.basis();
        Self {
            view_proj: c.view_proj().to_cols_array_2d(),
            eye: c.eye().extend(1.0).into(),
            right: right.extend(0.0).into(),
            up: up.extend(0.0).into(),
        }
    }
}

const SPHERE_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4];

/// Explicit offsets: [`Cylinder`] pads after `end` to keep the colours aligned.
const CYLINDER_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    attribute(wgpu::VertexFormat::Float32x3, 0, 0),
    attribute(wgpu::VertexFormat::Float32, 12, 1),
    attribute(wgpu::VertexFormat::Float32x3, 16, 2),
    attribute(wgpu::VertexFormat::Float32x4, 32, 3),
    attribute(wgpu::VertexFormat::Float32x4, 48, 4),
];

const fn attribute(
    format: wgpu::VertexFormat,
    offset: u64,
    location: u32,
) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format,
        offset,
        shader_location: location,
    }
}

fn pipeline(
    rs: &egui_wgpu::RenderState,
    layout: &wgpu::PipelineLayout,
    label: &str,
    source: &str,
    vertices: wgpu::VertexBufferLayout<'_>,
) -> wgpu::RenderPipeline {
    let shader = rs
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    rs.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertices)],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: rs.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // Impostors: four corners per instance.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                // Same mapping eframe uses for its own depth attachment, so
                // formats can't drift.
                format: egui_wgpu::depth_format_from_bits(super::DEPTH_BITS, 0)
                    .unwrap_or(wgpu::TextureFormat::Depth24Plus),
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
}

/// Grow-only GPU buffer of instances.
#[derive(Default)]
struct Instances {
    buffer: Option<wgpu::Buffer>,
    count: u32,
}

impl Instances {
    fn upload<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        items: &[T],
    ) {
        self.count = items.len() as u32;
        let bytes: &[u8] = bytemuck::cast_slice(items);
        if bytes.is_empty() {
            return;
        }
        // Reallocated only when a frame needs more room.
        if !self
            .buffer
            .as_ref()
            .is_some_and(|b| b.size() >= bytes.len() as u64)
        {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes.len().next_power_of_two() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytes);
        }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'static>, pipeline: &wgpu::RenderPipeline) {
        if let Some(buffer) = self.buffer.as_ref().filter(|_| self.count > 0) {
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..4, 0..self.count);
        }
    }
}

struct Resources {
    camera: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    spheres: (wgpu::RenderPipeline, Instances),
    cylinders: (wgpu::RenderPipeline, Instances),
    /// Key of the scene in the instance buffers, which every viewer shares.
    uploaded: Option<egui::Id>,
}

/// Sets up pipelines once; false if the app runs without wgpu.
pub fn register(cc: &eframe::CreationContext<'_>) -> bool {
    let Some(rs) = &cc.wgpu_render_state else {
        return false;
    };
    let device = &rs.device;
    let camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let instanced =
        |stride: usize, attributes: &'static [wgpu::VertexAttribute]| wgpu::VertexBufferLayout {
            array_stride: stride as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes,
        };
    let spheres = pipeline(
        rs,
        &layout,
        "spheres",
        include_str!("spheres.wgsl"),
        instanced(std::mem::size_of::<Sphere>(), &SPHERE_ATTRIBUTES),
    );
    let cylinders = pipeline(
        rs,
        &layout,
        "capsules",
        include_str!("capsules.wgsl"),
        instanced(std::mem::size_of::<Cylinder>(), &CYLINDER_ATTRIBUTES),
    );
    rs.renderer.write().callback_resources.insert(Resources {
        camera,
        bind_group,
        spheres: (spheres, Instances::default()),
        cylinders: (cylinders, Instances::default()),
        uploaded: None,
    });
    true
}

/// One frame's draw calls, painted inside the viewer's rect.
pub struct Draw {
    pub camera: Camera,
    /// Uploaded only when the GPU holds another key: an unchanged scene costs no
    /// copy, and a frame egui discards can't leave the buffers stale.
    pub scene: (egui::Id, Arc<Scene>),
}

impl egui_wgpu::CallbackTrait for Draw {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(r) = resources.get_mut::<Resources>() {
            let camera = CameraUniform::from(&self.camera);
            queue.write_buffer(&r.camera, 0, bytemuck::bytes_of(&camera));
            let (key, scene) = &self.scene;
            if r.uploaded != Some(*key) {
                r.spheres.1.upload(device, queue, &scene.spheres);
                r.cylinders.1.upload(device, queue, &scene.cylinders);
                r.uploaded = Some(*key);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(r) = resources.get::<Resources>() {
            pass.set_bind_group(0, &r.bind_group, &[]);
            for (pipeline, instances) in [&r.spheres, &r.cylinders] {
                instances.draw(pass, pipeline);
            }
        }
    }
}
