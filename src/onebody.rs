use crate::conditions::SolutionConditions;
use crate::structures::Structures;
use duello::archive::CgBody;

/// Home of the inputs every tab shares: structures and solution conditions.
/// The central area is kept for purrette's single-molecule analysis.
pub struct OnebodyTab;

impl OnebodyTab {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        structures: &mut Structures,
        conditions: &mut SolutionConditions,
        scanned_body: Option<&CgBody>,
    ) {
        egui::Panel::left("onebody_inputs")
            .min_size(220.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.strong("Structures");
                    ui.add_space(2.0);
                    structures.ui(ui);
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(2.0);
                    ui.strong("Solution conditions")
                        .on_hover_text("Shared by every tab");
                    ui.add_space(2.0);
                    conditions.ui(ui);
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new("Free-salt estimate")
                        .default_open(false)
                        .show(ui, |ui| conditions.estimator_ui(ui, scanned_body))
                        .header_response
                        .on_hover_text(
                            "Ionic strength from Donnan equilibrium with a salt reservoir, \
                            for net-charged, crowded proteins",
                        );
                });
            });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label("Single-molecule analysis is coming soon.")
                    .on_hover_text("Planned: titration and charge properties of one molecule");
            });
        });
    }
}
