//! 3D view of spheres and capsules, copied from purrette and trimmed to
//! what the Manybody box needs. Callers hand over a [`Scene`] each frame;
//! wgpu, the camera and input stay in here.

use std::sync::Arc;

mod camera;
mod gesture;
mod gpu;

use camera::Camera;

/// Depth buffer the window must be created with; impostors write exact depth into it.
pub const DEPTH_BITS: u8 = 24;

pub type Rgba = [f32; 4];

/// A sphere instance, laid out for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: Rgba,
}

/// A capsule instance, laid out for the GPU; each half takes its end's colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cylinder {
    pub start: [f32; 3],
    pub radius: f32,
    pub end: [f32; 3],
    /// Keeps the colours 16-byte aligned for the GPU.
    pub _pad: f32,
    pub start_color: Rgba,
    pub end_color: Rgba,
}

/// A triangle-mesh vertex, laid out for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: Rgba,
}

/// Opaque triangles, drawn both sides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    /// Three per triangle, into `vertices`.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// One draw call takes every mesh in a scene.
    pub fn append(&mut self, other: &Mesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices
            .extend(other.indices.iter().map(|i| i + offset));
    }
}

/// (centre, radius) of a sphere enclosing a scene.
pub type Bounds = ([f32; 3], f32);

/// What to draw this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub cylinders: Vec<Cylinder>,
    pub mesh: Mesh,
    pub bounds: Bounds,
}

/// Golden-angle hue steps keep any two nearby indices far apart in colour,
/// however many there are.
pub fn categorical_color(index: usize) -> Rgba {
    const GOLDEN_TURN: f32 = 0.381_966;
    let hue = (index as f32 * GOLDEN_TURN).fract();
    // The shaders write display values, so convert out of egui's linear space.
    egui::ecolor::Hsva::new(hue, 0.6, 0.9, 1.0)
        .to_srgba_unmultiplied()
        .map(|c| f32::from(c) / 255.0)
}

/// Red for negative, blue for positive, saturating at one charge, so a bead's sign
/// reads at a glance; Onebody's surface potential and Twobody's poses use the same map.
pub fn charge_color(z: f64) -> Rgba {
    cgkitten::coloring::diverging(z)
}

/// Set once pipelines exist; without wgpu the view says so instead.
fn gpu_id() -> egui::Id {
    egui::Id::unique("viewer_gpu")
}

/// Builds the GPU pipelines; call once at startup.
pub fn register(cc: &eframe::CreationContext<'_>) {
    let ready = gpu::register(cc);
    cc.egui_ctx.data_mut(|d| d.insert_temp(gpu_id(), ready));
}

pub struct Viewer {
    camera: Camera,
    /// What the camera was last fitted to.
    bounds: Option<Bounds>,
    viewport: [f32; 2],
    /// Once the user moves the camera, only a double-click or [`Self::reset`]
    /// refits it, so changing scenes (a running simulation, a re-titrated
    /// molecule) don't fight their view.
    touched: bool,
    /// The last scene built, by key; the GPU side uploads it when it holds another.
    scene: Option<(egui::Id, Arc<Scene>)>,
}

impl Default for Viewer {
    fn default() -> Self {
        Self {
            camera: Camera::new([1.0, 1.0]),
            bounds: None,
            viewport: [1.0, 1.0],
            touched: false,
            scene: None,
        }
    }
}

const HELP: &str = "Drag to orbit; right- or shift-drag to pan; scroll or pinch to zoom; \
                    twist to roll; double-click to reset the view";

impl Viewer {
    /// Fills the available space with the scene `build` makes. `build` runs,
    /// and the GPU gets new instances, only when `key` changes.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        key: impl std::hash::Hash + std::fmt::Debug,
        build: impl FnOnce() -> Scene,
    ) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        if !ready(ui, rect) {
            return;
        }
        // Salted with where the viewer sits, as the GPU buffers are shared.
        let key = ui.scope_id().with(key);
        if self.scene.as_ref().map(|s| s.0) != Some(key) {
            self.scene = Some((key, Arc::new(build())));
        }
        let scene = self.scene.clone().expect("set above");
        if response.double_clicked() {
            self.reset();
        }
        self.fit(scene.1.bounds, [rect.width(), rect.height()]);
        let gesture = gesture::gesture(ui, &response);
        self.touched |= gesture != gesture::Gesture::default();
        gesture::apply(&mut self.camera, gesture);
        let draw = gpu::Draw {
            camera: self.camera,
            scene,
        };
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(rect, draw));
        help_badge(ui, rect);
    }

    /// The default view, fitted to the next scene; for when the scene shows
    /// something else entirely.
    pub fn reset(&mut self) {
        self.camera = Camera::new(self.viewport);
        self.touched = false;
        self.bounds = None;
    }

    /// Follows new bounds and resizes until the user moves the camera.
    fn fit(&mut self, bounds: Bounds, viewport: [f32; 2]) {
        let changed = self.viewport != viewport || self.bounds != Some(bounds);
        self.viewport = viewport;
        self.camera.set_viewport(viewport);
        if changed && !self.touched {
            self.camera.fit(bounds.0, bounds.1);
        }
        self.bounds = Some(bounds);
    }
}

/// Paints the background; false while egui sizes the window or without
/// wgpu, when there is nothing to draw.
fn ready(ui: &egui::Ui, rect: egui::Rect) -> bool {
    if ui.is_sizing_pass() {
        return false;
    }
    ui.painter()
        .rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let gpu = ui.data(|d| d.get_temp(gpu_id()).unwrap_or(false));
    if !gpu {
        centered_text(ui, rect, "The 3D view needs WebGPU");
    }
    gpu
}

/// A corner "?" that explains the controls, so the help doesn't pop up
/// over the scene while the user works in it.
fn help_badge(ui: &mut egui::Ui, rect: egui::Rect) {
    let badge = egui::Rect::from_min_size(
        rect.right_top() + egui::vec2(-26.0, 4.0),
        egui::vec2(22.0, 22.0),
    );
    ui.put(badge, egui::Label::new("?").sense(egui::Sense::hover()))
        .on_hover_text(HELP);
}

fn centered_text(ui: &egui::Ui, rect: egui::Rect, text: &str) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(15.0),
        ui.visuals().weak_text_color(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_colour_shows_the_sign_and_saturates_at_one_charge() {
        let [r, g, b, a] = charge_color(0.0);
        assert!(r == g && g == b && a == 1.0, "neutral is grey");
        let [r, _, b, _] = charge_color(-0.5);
        assert!(r > b, "negative is red");
        let [r, _, b, _] = charge_color(0.5);
        assert!(b > r, "positive is blue");
        assert_eq!(charge_color(3.0), charge_color(1.0));
        assert_eq!(charge_color(-3.0), charge_color(-1.0));
        assert_ne!(charge_color(0.5), charge_color(1.0));
    }
}
