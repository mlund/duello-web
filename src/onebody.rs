/// Single-protein analysis. Empty for now; purrette's titration goes behind
/// this seam later without touching `app.rs`.
pub struct OnebodyTab;

impl OnebodyTab {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label("Single-molecule analysis is coming soon.")
                    .on_hover_text("Planned: titration and charge properties of one molecule");
            });
        });
    }
}
