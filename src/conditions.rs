//! The solution every tab works in; set in Onebody, read elsewhere.

use crate::widgets::{grid, row};
use duello::web_api::ScanSettings;

#[derive(Clone, Debug, PartialEq)]
pub struct SolutionConditions {
    /// K
    pub temperature: f64,
    pub ph: f64,
    /// mol/L
    pub ionic_strength: f64,
}

impl Default for SolutionConditions {
    fn default() -> Self {
        Self {
            temperature: 298.15,
            ph: 7.0,
            ionic_strength: 0.1,
        }
    }
}

const PH_TIP: &str = "Determines the average ionization state of titratable groups \
    and N/C terminal ends using Metropolis GCMC swap moves.";
const IONIC_STRENGTH_TIP: &str = "Sets the Debye screening length used in the Monte Carlo \
    titration of ionizable groups and in the intermolecular Yukawa energy. A scan \
    records it, and Manybody's reference sphere takes it from the scan.";
const TEMPERATURE_TIP: &str = "Affects the thermal energy used in Boltzmann averaging \
    and the dielectric constant of water.";

impl SolutionConditions {
    /// Takes the solution an imported scan was made in, so the tabs describe it.
    pub fn adopt(&mut self, settings: &ScanSettings) {
        self.temperature = settings.temperature;
        self.ph = settings.ph;
        self.ionic_strength = settings.ionic_strength;
    }

    /// Whether a scan was made in these conditions.
    pub fn matches(&self, settings: &ScanSettings) -> bool {
        let mut adopted = self.clone();
        adopted.adopt(settings);
        adopted == *self
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        grid("conditions_grid").show(ui, |ui| {
            row(ui, "pH", PH_TIP, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.ph)
                        .speed(0.1)
                        .range(0.0..=14.0),
                )
            });
            row(ui, "Ionic strength", IONIC_STRENGTH_TIP, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.ionic_strength)
                        .speed(0.01)
                        .suffix(" M")
                        .range(0.0..=10.0),
                )
            });
            row(ui, "Temperature", TEMPERATURE_TIP, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.temperature)
                        .speed(1.0)
                        .suffix(" K")
                        .range(1.0..=1000.0),
                )
            });
        });
    }

    /// Read-only, for tabs that use the conditions but don't own them.
    pub fn summary_ui(&self, ui: &mut egui::Ui) {
        let shown = |tip: &str| format!("{tip}\n\nSet in Onebody.");
        grid("conditions_summary").show(ui, |ui| {
            row(ui, "pH", &shown(PH_TIP), |ui| {
                ui.label(format!("{:.2}", self.ph))
            });
            row(ui, "Ionic strength", &shown(IONIC_STRENGTH_TIP), |ui| {
                ui.label(format!("{:.3} M", self.ionic_strength))
            });
            row(ui, "Temperature", &shown(TEMPERATURE_TIP), |ui| {
                ui.label(format!("{:.2} K", self.temperature))
            });
        });
    }
}
