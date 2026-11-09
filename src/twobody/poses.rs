//! The Poses window: the scan's lowest-energy states, drawn as beads.

use crate::viewer::{charge_color, Rgba, Scene, Sphere, Viewer};
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
        ui.weak("beads: red −, blue +; neutral sand (1), slate (2)")
            .on_hover_text(
                "Bead charge, full colour at ±1 e. Near-neutral beads are tinted by \
                 molecule: sand for molecule 1, slate for molecule 2.",
            );
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

/// Each molecule's neutral bead colour, warm and cool around the charge map's
/// grey, so the two bodies tell apart while red and blue still read as charge.
const NEUTRAL_TINTS: [Rgba; 2] = [[0.84, 0.78, 0.66, 1.0], [0.66, 0.74, 0.86, 1.0]];

/// The charge map with `neutral` in place of its grey; fades out as |z| → 1, so
/// a full charge looks the same in either molecule.
fn tinted_charge_color(z: f64, neutral: Rgba) -> Rgba {
    let end = charge_color(z.signum());
    let t = z.abs().min(1.0) as f32;
    std::array::from_fn(|i| (1.0 - t) * neutral[i] + t * end[i])
}

/// Both bodies as beads coloured by charge, framed about their joint centre.
fn scene(pose: &TablePose, bodies: &[CgBody; 2]) -> Scene {
    let placed = pose.place(bodies);
    let spheres: Vec<Sphere> = (placed.iter().zip(bodies).zip(NEUTRAL_TINTS))
        .flat_map(|((beads, body), neutral)| {
            (beads.iter().zip(&body.radii).zip(&body.charges)).map(move |((p, &r), &z)| Sphere {
                center: p.map(|x| x as f32),
                radius: r as f32,
                color: tinted_charge_color(z, neutral),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_beads_take_their_molecules_tint_and_full_charges_do_not() {
        let [a, b] = NEUTRAL_TINTS;
        assert_eq!(tinted_charge_color(0.0, a), a);
        assert_eq!(tinted_charge_color(0.0, b), b);
        for z in [-1.0, 1.0, 2.0] {
            assert_eq!(tinted_charge_color(z, a), charge_color(z));
            assert_eq!(tinted_charge_color(z, b), charge_color(z));
        }
    }
}
