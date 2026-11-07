//! The Poses window: the scan's lowest-energy states, drawn as beads.

use crate::compute::ComputeStatus;
use crate::viewer::{charge_color, Scene, Sphere, Viewer};
use crate::widgets::placeholder;
use duello::archive::{CgBody, ScanArchive};
use duello::poses::{lowest, TablePose};

/// How many states the picker offers.
const COUNT: usize = 10;

/// The ranked states and the 3D view of the picked one.
#[derive(Default)]
pub struct PosesView {
    /// Ranked once per result, as ranking walks the whole table.
    ranked: Option<Vec<TablePose>>,
    picked: usize,
    /// Bumped per result, so the viewer rebuilds its scene for a new one.
    result: u64,
    viewer: Viewer,
}

impl PosesView {
    /// `finished` is the scan to rank, None while there is none.
    pub fn ui(&mut self, ui: &mut egui::Ui, finished: Option<&ScanArchive>) {
        let Some(archive) = finished else {
            return placeholder(ui, "Available once the scan finishes");
        };
        let Self {
            ranked,
            picked,
            result,
            viewer,
        } = self;
        let poses = ranked.get_or_insert_with(|| lowest(archive, COUNT));
        let Some(pose) = poses.get(*picked) else {
            return placeholder(
                ui,
                "No stored orientations: every state is repulsive or isotropic",
            );
        };
        picker(ui, poses, picked);
        let bodies = &archive.bodies;
        viewer.ui(ui, (*result, *picked), || scene(pose, bodies));
    }

    /// For a new result: another scene entirely, so ranking, pick and view start over.
    pub fn invalidate(&mut self) {
        self.ranked = None;
        self.picked = 0;
        self.result += 1;
        self.viewer.reset();
    }
}

fn picker(ui: &mut egui::Ui, poses: &[TablePose], picked: &mut usize) {
    let tip = "The lowest-energy states stored in the scan's table, by pair energy U. \
               Neighbours in the list often sit in the same contact basin.";
    ui.horizontal(|ui| {
        ui.label("Pose").on_hover_text(tip);
        egui::ComboBox::from_id_salt("twobody_pose")
            .selected_text(label(*picked, &poses[*picked]))
            .show_ui(ui, |ui| {
                for (k, pose) in poses.iter().enumerate() {
                    ui.selectable_value(picked, k, label(k, pose));
                }
            })
            .response
            .on_hover_text(tip);
        ui.weak("beads: red −, blue +")
            .on_hover_text("Bead charge; full colour at ±1 e");
    });
}

fn label(k: usize, pose: &TablePose) -> String {
    format!(
        "#{}  U = {:.1} kT · R = {:.1} Å · ω = {:.0}°",
        k + 1,
        pose.energy,
        pose.r,
        pose.omega.to_degrees()
    )
}

/// Both bodies as beads coloured by charge, framed about their joint centre.
fn scene(pose: &TablePose, bodies: &[CgBody; 2]) -> Scene {
    let placed = pose.place(bodies);
    let spheres: Vec<Sphere> = (placed.iter().zip(bodies))
        .flat_map(|(beads, body)| {
            (beads.iter().zip(&body.radii).zip(&body.charges)).map(|((p, &r), &z)| Sphere {
                center: p.map(|x| x as f32),
                radius: r as f32,
                color: charge_color(z),
            })
        })
        .collect();
    let bounds = bounds(&spheres);
    Scene {
        spheres,
        bounds,
        ..Default::default()
    }
}

fn bounds(spheres: &[Sphere]) -> crate::viewer::Bounds {
    let n = spheres.len().max(1) as f32;
    let centre = spheres.iter().fold([0.0; 3], |c, s| {
        std::array::from_fn(|i| c[i] + s.center[i] / n)
    });
    let reach = |s: &Sphere| {
        let d: f32 = (0..3).map(|i| (s.center[i] - centre[i]).powi(2)).sum();
        d.sqrt() + s.radius
    };
    (centre, spheres.iter().map(reach).fold(1.0, f32::max))
}

/// Only a finished or imported scan has the whole table to rank.
pub fn poses_shown(status: &ComputeStatus, result: Option<&ScanArchive>) -> bool {
    matches!(status, ComputeStatus::Idle) && result.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poses_show_only_for_a_finished_scan() {
        let archive = crate::test_archive::archive();
        let running = ComputeStatus::Running {
            current_r: 1,
            total_r: 3,
        };
        assert!(!poses_shown(&running, None));
        assert!(!poses_shown(&running, Some(&archive)));
        assert!(!poses_shown(&ComputeStatus::Idle, None));
        assert!(poses_shown(&ComputeStatus::Idle, Some(&archive)));
    }
}
