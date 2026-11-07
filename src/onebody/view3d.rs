//! The Viewer window: the molecule in 3D, with the choices of what it shows.

use super::scene::{self, Colouring, Look, Show, ARROW_PER_DIPOLE};
use super::surface::{Field, Surface, SurfaceView};
use crate::viewer::Viewer;
use duello::molecule::Molecule;

#[derive(Default)]
pub struct MoleculeScene {
    viewer: Viewer,
    look: Look,
    surface: SurfaceView,
}

impl MoleculeScene {
    /// `generation` names `molecule`; it was titrated at `temperature` (K) and
    /// `ionic_strength` (mol/L), which also screen its potential.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        molecule: &Molecule,
        generation: u64,
        temperature: f64,
        ionic_strength: f64,
    ) {
        let mu = scene::dipole(molecule);
        controls(ui, &mut self.look, mu.length());
        let look = self.look;
        let surface = match look.show {
            Show::Surface(_) => {
                let surface =
                    self.surface
                        .surface(molecule, generation, temperature, ionic_strength);
                shown_or_warn(ui, surface)
            }
            Show::Beads => None,
        };
        let rect = ui.available_rect_before_wrap();
        self.viewer.ui(ui, (generation, look), || {
            scene::scene(molecule, mu, look, surface)
        });
        if let (Show::Surface(field), Some(surface)) = (look.show, surface) {
            legend(ui, rect, field, surface.scale(field));
        }
    }

    /// For a new molecule: the old view needn't suit it.
    pub fn reset_camera(&mut self) {
        self.viewer.reset();
    }
}

fn shown_or_warn<'a>(
    ui: &mut egui::Ui,
    surface: Result<&'a Surface, String>,
) -> Option<&'a Surface> {
    surface
        .inspect_err(|e| {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("No surface: {e}"));
        })
        .ok()
}

fn controls(ui: &mut egui::Ui, look: &mut Look, mu: f32) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Show")
            .on_hover_text("Beads, or the surface coloured by potential");
        for show in Show::ALL {
            ui.selectable_value(&mut look.show, show, show.label())
                .on_hover_text(show.tip());
        }
        if look.show == Show::Beads {
            ui.separator();
            ui.label("Colour")
                .on_hover_text("What the bead colours show");
            for c in Colouring::ALL {
                ui.selectable_value(&mut look.colouring, c, c.label())
                    .on_hover_text(c.tip());
            }
        }
        ui.separator();
        ui.checkbox(&mut look.dipole, "Dipole").on_hover_ui(|ui| {
            ui.label(format!(
                "Arrow along the mean charges' dipole from the mass centre, \
                 |μ| = {mu:.1} e·Å; it reaches {ARROW_PER_DIPOLE} Å per e·Å past \
                 the molecule"
            ));
        });
    });
}

/// The colour scale, in the view's lower-left corner.
fn legend(ui: &mut egui::Ui, rect: egui::Rect, field: Field, scale: f64) {
    let (text, tip) = field.legend(scale);
    let at = egui::Rect::from_min_size(
        rect.left_bottom() + egui::vec2(6.0, -22.0),
        egui::vec2(260.0, 18.0),
    );
    let label = egui::Label::new(egui::RichText::new(text).small()).sense(egui::Sense::hover());
    ui.put(at, label).on_hover_text(tip);
}
