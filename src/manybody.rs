//! Manybody tab: Metropolis MC of N copies of the loaded homo-dimer scan,
//! reporting I(q), S_eff(q), P(q) and run statistics.

use crate::app::save_file;
use duello::archive::ScanArchive;
use duello::nbody::{box_length, NBodyRun, Progress, RunSpec, Stage, RESIDUE_FORM_FACTORS};
use egui_plot::{Line, Plot, PlotPoints};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Wall time per UI frame spent stepping the run, so the page stays
/// responsive while the run makes progress.
const FRAME_BUDGET_MS: f64 = 30.0;

#[derive(Clone, Copy, PartialEq)]
enum View {
    StructureFactor,
    Intensity,
    FormFactor,
}

pub struct ManybodyTab {
    n_bodies: usize,
    millimolar: f64,
    copies: usize,
    run: Option<NBodyRun>,
    running: bool,
    /// Sweeps per UI frame, tuned to `FRAME_BUDGET_MS`.
    sweeps_per_frame: u64,
    progress: Option<Progress>,
    /// (sweep, U/N) while equilibrating.
    energy_trace: Vec<[f64; 2]>,
    view: View,
    error: Option<String>,
    /// What the current run was started with, so editing the inputs or
    /// loading another scan doesn't relabel its results.
    run_info: Option<RunInfo>,
}

#[derive(Clone, Copy)]
struct RunInfo {
    n_bodies: usize,
    millimolar: f64,
    salt_millimolar: f64,
}

impl Default for ManybodyTab {
    fn default() -> Self {
        Self {
            n_bodies: 200,
            millimolar: 5.0,
            copies: 4,
            run: None,
            running: false,
            sweeps_per_frame: 10,
            progress: None,
            energy_trace: Vec::new(),
            view: View::StructureFactor,
            error: None,
            run_info: None,
        }
    }
}

impl ManybodyTab {
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Advances a running simulation by one frame's budget; called every
    /// frame whichever tab is shown, so switching tabs doesn't pause it.
    pub fn tick(&mut self) {
        if self.running {
            self.advance();
        }
    }

    /// `archive` is the scan loaded in the Twobody tab, if any.
    pub fn ui(&mut self, ui: &mut egui::Ui, archive: Option<&ScanArchive>) {
        egui::Panel::left("manybody_inputs")
            .min_size(220.0)
            .show_inside(ui, |ui| self.ui_inputs(ui, archive));
        egui::Panel::bottom("manybody_stats")
            .min_size(80.0)
            .show_inside(ui, |ui| self.ui_stats(ui));
        egui::CentralPanel::default().show_inside(ui, |ui| self.ui_plot(ui));
    }

    fn ui_inputs(&mut self, ui: &mut egui::Ui, archive: Option<&ScanArchive>) {
        ui.strong("Many-body system");
        // Stop and errors don't depend on the loaded scan, which may have
        // changed since the run started.
        if self.running && ui.button("Stop").clicked() {
            self.stop();
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::RED, e);
        }
        let Some(archive) = archive.filter(|a| a.settings.homo_dimer) else {
            ui.label("Run or import a scan of two identical molecules in the Twobody tab first.");
            return;
        };
        let idle = !self.running;
        ui.add_enabled_ui(idle, |ui| {
            egui::Grid::new("manybody_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Molecules, N");
                    ui.add(egui::DragValue::new(&mut self.n_bodies).range(2..=2000));
                    ui.end_row();
                    ui.label("Concentration (mM)");
                    ui.add(
                        egui::DragValue::new(&mut self.millimolar)
                            .range(0.01..=200.0)
                            .speed(0.1),
                    );
                    ui.end_row();
                    ui.label("Independent copies");
                    ui.add(egui::DragValue::new(&mut self.copies).range(1..=16));
                    ui.end_row();
                });
        });
        let can_start = self.ui_system_info(ui, archive);
        if ui
            .add_enabled(idle && can_start, egui::Button::new("Run"))
            .clicked()
        {
            self.start(archive);
        }
    }

    /// Box, q range and screening for the chosen N and concentration; false
    /// if the box is too small for the table.
    fn ui_system_info(&self, ui: &mut egui::Ui, archive: &ScanArchive) -> bool {
        let l = box_length(self.n_bodies, self.millimolar * 1e-3);
        let needed = 2.0 * archive.table.rmax;
        ui.separator();
        ui.label(format!(
            "Box L = {l:.0} Å, q_min = {:.4} Å⁻¹",
            std::f64::consts::TAU / l
        ));
        ui.label(format!("Net charge = {:+.1} e", archive.net_charges().0));
        // Debye length for 1:1 salt in water at 25 °C.
        let debye = 3.04 / archive.settings.molarity.sqrt();
        ui.label(format!(
            "Salt {:.0} mM, Debye length ≈ {debye:.0} Å",
            archive.settings.molarity * 1e3
        ));
        let fits = l >= needed;
        if !fits {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                format!("Box below the table's {needed:.0} Å: increase N"),
            );
        }
        fits
    }

    fn start(&mut self, archive: &ScanArchive) {
        let mut spec = RunSpec::new(self.n_bodies, self.millimolar * 1e-3);
        spec.copies = self.copies;
        match NBodyRun::start(archive, spec, RESIDUE_FORM_FACTORS) {
            Ok(run) => {
                self.run = Some(run);
                self.run_info = Some(RunInfo {
                    n_bodies: self.n_bodies,
                    millimolar: self.millimolar,
                    salt_millimolar: archive.settings.molarity * 1e3,
                });
                self.running = true;
                self.error = None;
                self.progress = None;
                self.energy_trace.clear();
                self.sweeps_per_frame = 10;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn stop(&mut self) {
        if let Some(run) = &mut self.run {
            run.stop();
            // A zero-sweep step just reports the stopped state.
            self.progress = run.step(0).ok();
        }
        self.running = false;
    }

    /// One budgeted step, then retune the budget toward `FRAME_BUDGET_MS`.
    fn advance(&mut self) {
        let Some(run) = &mut self.run else { return };
        let was_equilibrating = matches!(
            self.progress.map(|p| p.stage),
            None | Some(Stage::Equilibrating)
        );
        let start = Instant::now();
        let progress = match run.step(self.sweeps_per_frame) {
            Ok(progress) => progress,
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                self.running = false;
                return;
            }
        };
        if was_equilibrating {
            // Includes the step on which equilibration ended.
            self.energy_trace
                .push([progress.sweep as f64, progress.energy_per_body]);
        }
        self.progress = Some(progress);
        if was_equilibrating && progress.stage != Stage::Equilibrating {
            // Production steps cost far more per sweep: start the budget over.
            self.sweeps_per_frame = 1;
            return;
        }
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let scale = (FRAME_BUDGET_MS / ms.max(0.1)).clamp(0.1, 2.0);
        self.sweeps_per_frame = ((self.sweeps_per_frame as f64 * scale) as u64).max(1);
    }

    fn ui_plot(&mut self, ui: &mut egui::Ui) {
        let Some(run) = &self.run else {
            ui.centered_and_justified(|ui| ui.label("Start a run to see I(q)."));
            return;
        };
        // Until production frames exist (also when stopped while
        // equilibrating), the energy trace is all there is to show.
        if self.progress.map_or(0, |p| p.frames) == 0 {
            Plot::new("energy_trace")
                .x_axis_label("Sweeps")
                .y_axis_label("U / N (kT)")
                .show(ui, |plot| {
                    plot.line(Line::new(
                        "U/N",
                        PlotPoints::from(self.energy_trace.clone()),
                    ));
                });
            return;
        }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view, View::StructureFactor, "S_eff(q)");
            ui.selectable_value(&mut self.view, View::Intensity, "I(q)");
            ui.selectable_value(&mut self.view, View::FormFactor, "P(q)");
        });
        let curve = run.curve();
        let points = |values: &[f64], log: bool| -> Vec<[f64; 2]> {
            curve
                .q
                .iter()
                .zip(values)
                .filter(|(_, v)| !log || **v > 0.0)
                .map(|(q, v)| [*q, if log { v.log10() } else { *v }])
                .collect()
        };
        let (label, series) = match self.view {
            View::StructureFactor => ("S_eff", points(&curve.s_eff, false)),
            View::Intensity => ("log₁₀ I", points(&curve.intensity, true)),
            View::FormFactor => ("log₁₀ P", points(&curve.form_factor, true)),
        };
        let band = (self.view == View::StructureFactor)
            .then_some(curve.s_eff_error.as_ref())
            .flatten()
            .map(|errors| {
                let shifted = |sign: f64| -> Vec<[f64; 2]> {
                    curve
                        .q
                        .iter()
                        .zip(&curve.s_eff)
                        .zip(errors)
                        .map(|((q, s), e)| [*q, s + sign * e])
                        .collect()
                };
                (shifted(1.0), shifted(-1.0))
            });
        Plot::new("manybody_plot")
            .x_axis_label("q (Å⁻¹)")
            .y_axis_label(label)
            .show(ui, |plot| {
                plot.line(Line::new(label, PlotPoints::from(series)).width(2.0_f32));
                if let Some((upper, lower)) = band {
                    plot.line(Line::new("+1σ", PlotPoints::from(upper)).width(0.5_f32));
                    plot.line(Line::new("−1σ", PlotPoints::from(lower)).width(0.5_f32));
                }
            });
    }

    fn ui_stats(&mut self, ui: &mut egui::Ui) {
        let (Some(run), Some(progress), Some(info)) = (&self.run, self.progress, self.run_info)
        else {
            return;
        };
        let stage = match progress.stage {
            Stage::Equilibrating => "equilibrating".to_string(),
            Stage::Production { equilibrated: true } => "production".to_string(),
            Stage::Production {
                equilibrated: false,
            } => "production (not equilibrated: sweep cap reached)".to_string(),
            Stage::Stopped => "stopped".to_string(),
        };
        ui.label(format!(
            "Run: N = {}, {:.2} mM protein, {:.0} mM salt. Stage: {stage}; sweep {} per copy; {} frames; U/N = {:.2} kT",
            info.n_bodies,
            info.millimolar,
            info.salt_millimolar,
            progress.sweep,
            progress.frames,
            progress.energy_per_body
        ));
        let stats = run.stats();
        let n = info.n_bodies as f64;
        let show = |x: Option<f64>| x.map_or("–".to_string(), |v| format!("{v:.3}"));
        // Production statistics are empty until the first production frame.
        let produced = (stats.energy.frames > 0).then_some(());
        ui.label(format!(
            "Acceptance: translate {}, rotate {}; ⟨Δr²⟩ {} Å², ⟨Δθ²⟩ {} rad²",
            show(stats.moves.translate.acceptance()),
            show(stats.moves.rotate.acceptance()),
            show(stats.moves.translate.mean_square_displacement()),
            show(stats.moves.rotate.mean_square_displacement()),
        ));
        ui.label(format!(
            "⟨U⟩/N = {} kT, Var(U)/N = {} kT²",
            show(produced.map(|_| stats.energy.mean / n)),
            show(produced.map(|_| stats.energy.variance / n)),
        ));
        if !progress.recording {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Trajectory reached its size cap; recording stopped",
            );
        }
        let (mut csv, mut trajectory) = (false, false);
        ui.horizontal(|ui| {
            csv = ui.button("Curve (.csv)").clicked();
            trajectory = ui.button("Trajectory (.traj)").clicked();
        });
        // After the row: saving reports errors through `&mut self`, which
        // the borrow of `run` above rules out inside it.
        let saved = if csv {
            save_file(run.report_csv().as_bytes(), "manybody.csv").map_err(anyhow::Error::from)
        } else if trajectory {
            run.trajectory_bytes()
                .and_then(|bytes| Ok(save_file(&bytes, "manybody.traj")?))
        } else {
            Ok(())
        };
        match saved {
            Err(e) => self.error = Some(format!("Save failed: {e:#}")),
            Ok(()) if csv || trajectory => self.error = None,
            Ok(()) => {}
        }
    }
}
