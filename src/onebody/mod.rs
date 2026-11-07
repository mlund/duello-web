//! Home of the inputs every tab shares (structures, force field, solution
//! conditions) and the analysis of the selected structure.

use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use crate::structures::Structures;
use crate::widgets::section;
use crate::windows::{self, Slot};
use analysis::MoleculeView;
use duello::archive::CgBody;

mod analysis;
mod panes;
mod scene;
mod table;

#[derive(Default)]
pub struct OnebodyTab {
    view: MoleculeView,
}

impl OnebodyTab {
    /// `scanned_body` is a homo-dimer scan's molecule, preferred by the free-salt
    /// estimate as Manybody uses it.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        structures: &mut Structures,
        conditions: &mut SolutionConditions,
        forcefield: &mut ForceField,
        scanned_body: Option<&CgBody>,
    ) {
        let selected = structures
            .selected()
            .and_then(|id| Some((id, structures.get(id)?)));
        // Not while a value is dragged, so a drag restarts once, on release.
        let settled = ui.ctx().dragged_id().is_none();
        if self.view.tick(selected, forcefield, conditions, settled) {
            ui.ctx().request_repaint();
        }
        egui::Panel::left("onebody_inputs")
            .min_size(220.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_inputs(ui, structures, conditions, forcefield, scanned_body)
                });
            });
        egui::CentralPanel::default().show(ui, |ui| {
            for pane in Pane::ALL {
                windows::show(ui, pane, |ui| self.view.ui_pane(ui, pane));
            }
        });
    }

    fn ui_inputs(
        &mut self,
        ui: &mut egui::Ui,
        structures: &mut Structures,
        conditions: &mut SolutionConditions,
        forcefield: &mut ForceField,
        scanned_body: Option<&CgBody>,
    ) {
        ui.strong("Structures")
            .on_hover_text("Click a name to analyse it in the windows");
        ui.add_space(2.0);
        structures.ui(ui);
        section(ui);
        ui.strong("Force field")
            .on_hover_text("Shared by every tab");
        ui.add_space(2.0);
        forcefield.ui(ui);
        section(ui);
        ui.strong("Solution conditions")
            .on_hover_text("Shared by every tab");
        ui.add_space(2.0);
        conditions.ui(ui);
        ui.add_space(4.0);
        let body = match scanned_body {
            Some(body) => Some((body, "the Twobody scan's molecule")),
            None => self
                .view
                .body()
                .map(|body| (body, "the selected structure")),
        };
        egui::CollapsingHeader::new("Free-salt estimate")
            .default_open(false)
            .show(ui, |ui| conditions.estimator_ui(ui, body))
            .header_response
            .on_hover_text(
                "Ionic strength from Donnan equilibrium with a salt reservoir, \
                for net-charged, crowded proteins",
            );
        section(ui);
        self.view.status_ui(ui);
    }
}

/// The analysis windows, in tiling order.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pane {
    Titration,
    Multipole,
    Viewer,
    Atoms,
}

impl Pane {
    const ALL: [Pane; 4] = [Pane::Titration, Pane::Multipole, Pane::Viewer, Pane::Atoms];
}

impl windows::Pane for Pane {
    const GRID: [usize; 2] = [2, 2];

    fn title(self) -> &'static str {
        match self {
            Pane::Multipole => "Multipole",
            Pane::Titration => "Titration",
            Pane::Viewer => "Viewer",
            Pane::Atoms => "Atoms",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Pane::Multipole => "Charge, dipole and quadrupole moments and their fluctuations",
            Pane::Titration => {
                "Charge, dipole, quadrupole or capacitance over pH: Monte Carlo and ideal"
            }
            Pane::Viewer => "3D view of the coarse-grained beads",
            Pane::Atoms => "Every bead with its charge and force-field parameters",
        }
    }

    /// Titration and moments above, viewer and bead table below.
    fn slot(self) -> Slot {
        let (col, row, span) = match self {
            Pane::Titration => (0, 0, 1),
            Pane::Multipole => (1, 0, 1),
            Pane::Viewer => (0, 1, 1),
            Pane::Atoms => (1, 1, 1),
        };
        Slot { col, row, span }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn panes_tile_the_canvas() {
        crate::windows::assert_tiled(&super::Pane::ALL);
    }
}
