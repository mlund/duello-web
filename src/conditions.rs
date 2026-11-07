//! The solution every tab works in; set in Onebody, read elsewhere.

use crate::widgets::{grid, row};
use duello::archive::{CgBody, FreeSaltEstimate, Session};
use duello::nbody::HsyFluid;
use duello::solution::{free_salt, protein_volume, FreeSalt};
use duello::web_api::ScanSettings;

#[derive(Clone, Debug, PartialEq)]
pub struct SolutionConditions {
    /// K
    pub temperature: f64,
    pub ph: f64,
    /// mol/L
    pub ionic_strength: f64,
    /// Sets Manybody's box and the free-salt estimate.
    pub protein_millimolar: f64,
    /// What a free-salt estimate assumed when it set the ionic strength;
    /// a hand edit of the ionic strength clears it.
    pub estimate: Option<FreeSaltEstimate>,
    estimator: EstimatorInputs,
}

/// Free-salt estimator inputs; Z and molar mass are used only without a
/// homo-dimer scan to take them from.
#[derive(Clone, Debug, PartialEq)]
struct EstimatorInputs {
    /// mol/L
    reservoir: f64,
    charge: f64,
    /// g/mol
    molar_mass: f64,
}

impl Default for SolutionConditions {
    fn default() -> Self {
        Self {
            temperature: 298.15,
            ph: 7.0,
            ionic_strength: 0.1,
            protein_millimolar: 5.0,
            estimate: None,
            // Lysozyme at pH 7, as an example to edit.
            estimator: EstimatorInputs {
                reservoir: 0.1,
                charge: 8.0,
                molar_mass: 14_300.0,
            },
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
const PROTEIN_TIP: &str = "Protein concentration: with N it sets Manybody's box, and the \
    free-salt estimate counts its counterions and volume.";
const RESERVOIR_TIP: &str = "Salt of a reservoir (e.g. the dialysis buffer) in Donnan \
    equilibrium with the solution. Net-charged proteins bring counterions and take up \
    volume, pushing co-ions out; the co-ions left in the free volume V(1 − η) are the \
    estimate. Counterions are deliberately not counted: they stay near the proteins. \
    For crowded proteins this is a better default than the buffer value, not an exact fix.";
const CHARGE_TIP: &str = "The net charge depends on the ionic strength through \
    titration: after Use, re-scan in Twobody and press Use again until it settles.";

impl SolutionConditions {
    /// Takes the solution an imported scan was made in, so the tabs describe it.
    pub fn adopt(&mut self, settings: &ScanSettings) {
        self.temperature = settings.temperature;
        self.ph = settings.ph;
        self.ionic_strength = settings.ionic_strength;
    }

    /// mol/L, as Manybody and the estimate use it.
    pub fn protein_molarity(&self) -> f64 {
        self.protein_millimolar * 1e-3
    }

    /// Everything a saved session knows of the solution: the scan's
    /// conditions, its estimate, and the protein concentration Manybody's
    /// inputs held at export.
    pub fn restore(&mut self, settings: &ScanSettings, session: &Session) {
        self.adopt(settings);
        self.estimate = session.estimate;
        if let Some(e) = session.estimate {
            self.estimator.reservoir = e.reservoir;
        }
        if let Some(run) = &session.run {
            self.protein_millimolar = run.molarity * 1e3;
        }
    }

    /// The estimate to save with a scan: only if it set the ionic strength
    /// the scan was made at.
    pub fn estimate_for(&self, settings: &ScanSettings) -> Option<FreeSaltEstimate> {
        self.estimate
            .filter(|_| same(settings.ionic_strength, self.ionic_strength))
    }

    /// Why a scan no longer reflects these conditions, if it doesn't.
    pub fn stale(&self, settings: &ScanSettings) -> Option<String> {
        let mut reasons = Vec::new();
        let mut differs = |name, scan: f64, now: f64, unit| {
            if !same(scan, now) {
                reasons.push(format!(
                    "{name} {scan:.3}{unit} in the scan, {now:.3}{unit} now"
                ));
            }
        };
        differs(
            "ionic strength",
            settings.ionic_strength,
            self.ionic_strength,
            " M",
        );
        differs("pH", settings.ph, self.ph, "");
        differs("temperature", settings.temperature, self.temperature, " K");
        if let Some(e) = self.estimate {
            let assumed = e.protein_molarity * 1e3;
            if !same(assumed, self.protein_millimolar) {
                reasons.push(format!(
                    "the free-salt estimate assumed {assumed:.2} mM protein, not {:.2} mM: \
                     press Use again in Onebody",
                    self.protein_millimolar
                ));
            }
        }
        (!reasons.is_empty()).then(|| reasons.join("; "))
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
                let edit = ui.add(
                    egui::DragValue::new(&mut self.ionic_strength)
                        .speed(0.01)
                        .suffix(" M")
                        .range(0.0..=10.0),
                );
                if edit.changed() {
                    self.estimate = None;
                }
                edit
            });
            row(ui, "Temperature", TEMPERATURE_TIP, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.temperature)
                        .speed(1.0)
                        .suffix(" K")
                        .range(1.0..=1000.0),
                )
            });
            row(ui, "Protein", PROTEIN_TIP, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.protein_millimolar)
                        .range(0.01..=200.0)
                        .speed(0.1)
                        .suffix(" mM"),
                )
            });
        });
    }

    /// Free salt for the protein `body` (a homo-dimer scan's molecule) or,
    /// without one, a hand-entered Z and molar mass; "Use" adopts it.
    pub fn estimator_ui(&mut self, ui: &mut egui::Ui, body: Option<&CgBody>) {
        let (charge, molar_mass) = self.estimator_grid(ui, body);
        let protein_molarity = self.protein_molarity();
        let reservoir = self.estimator.reservoir;
        let volume = protein_volume(molar_mass);
        let salt = match free_salt(reservoir, charge, protein_molarity, volume) {
            Ok(salt) => salt,
            Err(e) => {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("{e:#}"));
                return;
            }
        };
        let sphere = HsyFluid::protein(
            molar_mass,
            charge,
            protein_molarity,
            salt.molarity,
            self.temperature,
        );
        show_free_salt(ui, &salt, sphere.debye_length);
        if ui
            .button("Use")
            .on_hover_text("Set the ionic strength to this free salt")
            .clicked()
        {
            self.ionic_strength = salt.molarity;
            self.estimate = Some(FreeSaltEstimate {
                reservoir,
                protein_molarity,
            });
        }
    }

    /// The estimator's inputs; returns Z and molar mass in use.
    fn estimator_grid(&mut self, ui: &mut egui::Ui, body: Option<&CgBody>) -> (f64, f64) {
        let inputs = &mut self.estimator;
        grid("estimator_grid")
            .show(ui, |ui| {
                row(ui, "Reservoir salt", RESERVOIR_TIP, |ui| {
                    ui.add(
                        egui::DragValue::new(&mut inputs.reservoir)
                            .speed(0.01)
                            .suffix(" M")
                            .range(0.0..=10.0),
                    )
                });
                match body {
                    Some(body) => scanned_body_rows(ui, body),
                    None => inputs.hand_entry_rows(ui),
                }
            })
            .inner
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

impl EstimatorInputs {
    /// Z and molar mass typed in, for when no homo-dimer scan provides them.
    fn hand_entry_rows(&mut self, ui: &mut egui::Ui) -> (f64, f64) {
        let tip = format!("{CHARGE_TIP}\n\nNo homo-dimer scan loaded: enter it by hand.");
        row(ui, "Net charge Z", &tip, |ui| {
            ui.add(
                egui::DragValue::new(&mut self.charge)
                    .speed(0.1)
                    .suffix(" e"),
            )
        });
        let tip = "Sets the protein volume; enter by hand without a scan";
        row(ui, "Molar mass", tip, |ui| {
            ui.add(
                egui::DragValue::new(&mut self.molar_mass)
                    .speed(100.0)
                    .range(100.0..=1e7)
                    .suffix(" g/mol"),
            )
        });
        (self.charge, self.molar_mass)
    }
}

/// Z and molar mass of the scanned molecule, shown read-only.
fn scanned_body_rows(ui: &mut egui::Ui, body: &CgBody) -> (f64, f64) {
    let tip = format!("{CHARGE_TIP}\n\nFrom the Twobody scan's molecule.");
    row(ui, "Net charge Z", &tip, |ui| {
        ui.label(format!("{:+.1} e", body.net_charge()))
    });
    let tip = "From the Twobody scan's molecule; sets its volume";
    row(ui, "Molar mass", tip, |ui| {
        ui.label(format!("{:.0} g/mol", body.mass()))
    });
    (body.net_charge(), body.mass())
}

/// Equal up to unit round trips such as mM → M → mM.
fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

fn show_free_salt(ui: &mut egui::Ui, salt: &FreeSalt, debye_length: f64) {
    ui.label(format!("Volume fraction η = {:.4}", salt.volume_fraction))
        .on_hover_text("Protein volume per solution volume, from the partial specific volume");
    ui.label(format!("Free salt = {:.1} mM", salt.molarity * 1e3))
        .on_hover_text(RESERVOIR_TIP);
    ui.label(format!("Debye length ≈ {debye_length:.1} Å"))
        .on_hover_text("Screening length at the free salt and this temperature");
}
