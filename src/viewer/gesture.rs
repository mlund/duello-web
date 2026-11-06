//! Pointer and touch input as one camera step, so mouse, trackpad and
//! multitouch share a single, testable path to the camera.

use super::Camera;

/// One frame's worth of view change, in points and radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gesture {
    pub orbit: [f32; 2],
    pub pan: [f32; 2],
    /// Factor; > 1 moves closer, 1 leaves the distance.
    pub zoom: f32,
    pub roll: f32,
}

/// No change: zoom is a factor, so its identity is 1, not 0.
impl Default for Gesture {
    fn default() -> Self {
        Self {
            orbit: [0.0; 2],
            pan: [0.0; 2],
            zoom: 1.0,
            roll: 0.0,
        }
    }
}

pub fn apply(camera: &mut Camera, g: Gesture) {
    camera.rotate(g.orbit);
    camera.roll(g.roll);
    camera.pan(g.pan);
    camera.zoom(g.zoom);
}

/// This frame's gesture over `response`. Two fingers pan, pinch zooms and
/// twist rolls (trackpad pinch and rotation arrive the same way); with one
/// pointer, as in purrette: left-drag orbits, right- or shift-drag pans,
/// scroll zooms.
pub fn gesture(ui: &egui::Ui, response: &egui::Response) -> Gesture {
    if !response.hovered() && !response.dragged() {
        return Gesture::default();
    }
    let (touch, zoom, roll, shift, scroll) = ui.input(|i| {
        (
            i.multi_touch(),
            i.zoom_delta(),
            i.rotation_delta(),
            i.modifiers.shift,
            i.smooth_scroll_delta.y,
        )
    });
    let mut g = Gesture {
        zoom,
        roll,
        ..Default::default()
    };
    if let Some(touch) = touch {
        g.pan = touch.translation_delta.into();
        return g;
    }
    g.zoom *= (scroll * 0.002).exp();
    let drag = response.drag_delta().into();
    let pan = response.dragged_by(egui::PointerButton::Secondary) || (response.dragged() && shift);
    if pan {
        g.pan = drag;
    } else if response.dragged_by(egui::PointerButton::Primary) {
        g.orbit = drag;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        let mut camera = Camera::new([400.0, 300.0]);
        camera.fit([0.0; 3], 10.0);
        camera.rotate([30.0, -20.0]);
        camera
    }

    /// On-screen length of a 1 Å segment through the target, across the view.
    fn apparent_size(camera: &Camera) -> f32 {
        let (right, _) = camera.basis();
        let half = (0.5 * right).to_array();
        let a = camera.project(half).unwrap();
        let b = camera.project((-0.5 * right).to_array()).unwrap();
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    }

    /// At the target's depth, size on screen goes as 1/distance.
    #[test]
    fn zoom_two_halves_the_distance() {
        let mut zoomed = camera();
        apply(
            &mut zoomed,
            Gesture {
                zoom: 2.0,
                ..Default::default()
            },
        );
        let ratio = apparent_size(&zoomed) / apparent_size(&camera());
        assert!((ratio - 2.0).abs() < 1e-4, "{ratio}");

        let mut still = camera();
        apply(&mut still, Gesture::default());
        assert!(still.view_proj().abs_diff_eq(camera().view_proj(), 1e-6));
    }

    /// Panning moves the view by the drag: the old target, at the depth the
    /// pan is scaled for, slides exactly one point right and keeps its size.
    #[test]
    fn pan_slides_the_view_without_zooming() {
        let mut panned = camera();
        apply(
            &mut panned,
            Gesture {
                pan: [1.0, 0.0],
                ..Default::default()
            },
        );
        let before = camera().project([0.0; 3]).unwrap();
        let after = panned.project([0.0; 3]).unwrap();
        assert!(
            (after[0] - before[0] - 1.0).abs() < 1e-3,
            "{before:?} → {after:?}"
        );
        assert!(
            (after[1] - before[1]).abs() < 1e-3,
            "{before:?} → {after:?}"
        );
        let ratio = apparent_size(&panned) / apparent_size(&camera());
        assert!((ratio - 1.0).abs() < 1e-4, "{ratio}");
    }

    #[test]
    fn a_twist_rolls_the_camera() {
        let mut twisted = camera();
        apply(
            &mut twisted,
            Gesture {
                roll: 0.4,
                ..Default::default()
            },
        );
        let mut rolled = camera();
        rolled.roll(0.4);
        assert!(twisted.view_proj().abs_diff_eq(rolled.view_proj(), 1e-6));
    }
}
