//! Manybody tab: Metropolis MC of N copies of the loaded homo-dimer scan,
//! reporting S_com(q) against charged-sphere theory, optionally bead-level
//! I(q), P(q) and S_eff(q), and run statistics.

use crate::app::save_file;
use duello::archive::ScanArchive;
use duello::nbody::{
    box_length, NBodyRun, Progress, RunSpec, ScatteringCurve, Stage, RESIDUE_FORM_FACTORS,
};
use egui_plot::{Line, MarkerShape, Plot, PlotPoints, Points};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Wall time per UI frame spent stepping the run, so the page stays
/// responsive while the run makes progress.
const FRAME_BUDGET_MS: f64 = 30.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum View {
    StructureFactor,
    Intensity,
    FormFactor,
}

/// How a series is drawn: simulated curves solid, the theory dashed.
#[derive(Clone, Copy)]
enum Style {
    Primary,
    Secondary,
    Reference,
}

struct Series<'a> {
    name: &'static str,
    values: &'a [f64],
    /// Where copies give them.
    errors: Option<&'a [f64]>,
    style: Style,
}

impl<'a> Series<'a> {
    fn new(name: &'static str, values: &'a [f64], errors: Option<&'a [f64]>, style: Style) -> Self {
        Self {
            name,
            values,
            errors,
            style,
        }
    }
}

impl View {
    const ALL: [View; 3] = [View::StructureFactor, View::Intensity, View::FormFactor];

    fn label(self) -> &'static str {
        match self {
            View::StructureFactor => "S(q)",
            View::Intensity => "I(q)",
            View::FormFactor => "P(q)",
        }
    }

    /// I and P are per molecule, i.e. I/c up to contrast and molar mass.
    fn axis_label(self) -> &'static str {
        match self {
            View::StructureFactor => "S(q)",
            View::Intensity => "I(q)/c (relative)",
            View::FormFactor => "P(q) (relative)",
        }
    }

    /// I and P come from the bead sums, which only detailed runs do.
    fn offered(self, detailed: bool) -> bool {
        detailed || self == View::StructureFactor
    }

    /// S_com and its theory are cheap, so every run shows them; S_eff joins
    /// when beads are summed.
    fn curves(self, curve: &ScatteringCurve, detailed: bool) -> Vec<Series<'_>> {
        match self {
            View::StructureFactor => {
                let mut all = vec![Series::new(
                    "S_com",
                    &curve.s_com,
                    curve.s_com_error.as_deref(),
                    Style::Primary,
                )];
                if detailed {
                    all.push(Series::new(
                        "S_eff",
                        &curve.s_eff,
                        curve.s_eff_error.as_deref(),
                        Style::Secondary,
                    ));
                }
                if !curve.s_hsy.is_empty() {
                    all.push(Series::new(
                        "charged sphere (MPB-RMSA)",
                        &curve.s_hsy,
                        None,
                        Style::Reference,
                    ));
                }
                all
            }
            View::Intensity => vec![Series::new("I(q)", &curve.intensity, None, Style::Primary)],
            View::FormFactor => vec![Series::new(
                "P(q)",
                &curve.form_factor,
                None,
                Style::Primary,
            )],
        }
    }

    /// S is read linearly, I and P as log I vs q, the usual protein SAXS
    /// figure.
    fn default_log_axes(self) -> LogAxes {
        LogAxes {
            x: false,
            y: self != View::StructureFactor,
        }
    }
}

#[derive(Clone, Copy)]
struct LogAxes {
    x: bool,
    y: bool,
}

pub struct ManybodyTab {
    n_bodies: usize,
    millimolar: f64,
    copies: usize,
    detailed: bool,
    /// Lattice q-vectors per |q| bin for the bead sums.
    vectors_per_bin: usize,
    /// Starting (or, untuned, fixed) translation (Å) and rotation (rad).
    translation: f64,
    rotation: f64,
    tune_steps: bool,
    run: Option<NBodyRun>,
    running: bool,
    /// Sweeps per UI frame, tuned to `FRAME_BUDGET_MS`.
    sweeps_per_frame: u64,
    progress: Option<Progress>,
    /// (sweep, U/N) while equilibrating.
    energy_trace: Vec<[f64; 2]>,
    view: View,
    /// Indexed by `View as usize`.
    log_axes: [LogAxes; View::ALL.len()],
    /// Whether the energy plot tracks the recent range; see `recent_range`.
    follow_energy: bool,
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
    detailed: bool,
    tune_steps: bool,
}

impl Default for ManybodyTab {
    fn default() -> Self {
        let defaults = RunSpec::new(1, 1.0);
        Self {
            n_bodies: 200,
            millimolar: 5.0,
            copies: 4,
            detailed: false,
            vectors_per_bin: defaults.vectors_per_bin,
            translation: defaults.translation,
            rotation: defaults.rotation,
            tune_steps: true,
            run: None,
            running: false,
            sweeps_per_frame: 10,
            progress: None,
            energy_trace: Vec::new(),
            view: View::StructureFactor,
            log_axes: View::ALL.map(View::default_log_axes),
            follow_energy: true,
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
        // Run controls and errors don't depend on the loaded scan, which
        // may have changed since the run started.
        self.ui_run_controls(ui);
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
                    let defaults = RunSpec::new(1, 1.0);
                    ui.label("Detailed I(q)").on_hover_text(format!(
                        "Also sum over beads for I(q), P(q) and S_eff: a frame every {} \
                         sweeps instead of {}. S_com and, where it has a solution, the \
                         charged-sphere S(q) are always shown",
                        defaults.sweeps_per_frame, defaults.sweeps_per_check
                    ));
                    ui.checkbox(&mut self.detailed, "");
                    ui.end_row();
                    ui.label("q-vectors per bin").on_hover_text(
                        "Lattice q-vectors per 2π/L bin for the bead sums: cost grows with \
                         this; more gives smoother I(q) and S_eff",
                    );
                    ui.add_enabled(
                        self.detailed,
                        egui::DragValue::new(&mut self.vectors_per_bin).range(1..=500),
                    );
                    ui.end_row();
                });
            egui::CollapsingHeader::new("Monte Carlo")
                .default_open(false)
                .show(ui, |ui| self.ui_mc_inputs(ui));
        });
        let can_start = self.ui_system_info(ui, archive);
        if ui
            .add_enabled(idle && can_start, egui::Button::new("Run"))
            .clicked()
        {
            self.start(archive);
        }
    }

    /// Defaults suit most runs, hence collapsed.
    fn ui_mc_inputs(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("manybody_mc_grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Independent copies")
                    .on_hover_text("Error bars come from the spread between copies");
                ui.add(egui::DragValue::new(&mut self.copies).range(1..=16));
                ui.end_row();
                ui.label("Tune move steps").on_hover_text(
                    "During equilibration, find the steps that move bodies furthest per \
                         attempted move; frozen for production",
                );
                ui.checkbox(&mut self.tune_steps, "");
                ui.end_row();
                let start = if self.tune_steps { " (start)" } else { "" };
                ui.label(format!("Translation step{start} (Å)"));
                // Steps beyond L/2 add nothing (and are clamped).
                let half_box = box_length(self.n_bodies, self.millimolar * 1e-3) / 2.0;
                ui.add(
                    egui::DragValue::new(&mut self.translation)
                        .range(0.01..=half_box)
                        .speed(0.1),
                );
                ui.end_row();
                ui.label(format!("Rotation step{start} (rad)"));
                ui.add(
                    egui::DragValue::new(&mut self.rotation)
                        .range(0.0..=std::f64::consts::PI)
                        .speed(0.01),
                );
                ui.end_row();
            });
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
        spec.centres_only = !self.detailed;
        spec.vectors_per_bin = self.vectors_per_bin;
        spec.translation = self.translation;
        spec.rotation = self.rotation;
        spec.tune_steps = self.tune_steps;
        if !self.detailed {
            // Mass-centre frames cost little next to the sweeps between them.
            spec.sweeps_per_frame = spec.sweeps_per_check;
        }
        match NBodyRun::start(archive, spec, RESIDUE_FORM_FACTORS) {
            Ok(run) => {
                self.run = Some(run);
                self.run_info = Some(RunInfo {
                    n_bodies: self.n_bodies,
                    millimolar: self.millimolar,
                    salt_millimolar: archive.settings.molarity * 1e3,
                    detailed: self.detailed,
                    tune_steps: self.tune_steps,
                });
                self.running = true;
                self.error = None;
                self.progress = None;
                self.energy_trace.clear();
                self.follow_energy = true;
                self.sweeps_per_frame = 10;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// Pause keeps the run, unlike Stop; a paused run can also be
    /// replaced by pressing Run.
    fn ui_run_controls(&mut self, ui: &mut egui::Ui) {
        let stage = self.progress.map(|p| p.stage);
        let live = self.run.is_some() && stage != Some(Stage::Stopped);
        if !live {
            return;
        }
        ui.horizontal(|ui| {
            let pause = if self.running { "Pause" } else { "Resume" };
            if ui.button(pause).clicked() {
                self.running = !self.running;
            }
            if ui.button("Stop").clicked() {
                self.stop();
            }
        });
        let producing = matches!(stage, Some(Stage::Production { .. }));
        let reset = ui
            .add_enabled(producing, egui::Button::new("Reset statistics"))
            .on_hover_text("Discard production frames so far and keep sampling")
            .on_disabled_hover_text("Available once production starts");
        if reset.clicked() {
            self.reset_statistics();
        }
    }

    fn reset_statistics(&mut self) {
        let Some(run) = &mut self.run else { return };
        match run.reset_statistics() {
            Ok(progress) => self.progress = Some(progress),
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
        let color = curve_color(ui.visuals().dark_mode);
        // Until production frames exist (also when stopped while
        // equilibrating), the energy trace is all there is to show.
        if self.progress.map_or(0, |p| p.frames) == 0 {
            self.ui_energy_trace(ui, color);
            return;
        }
        let detailed = self.run_info.is_some_and(|i| i.detailed);
        if !self.view.offered(detailed) {
            self.view = View::StructureFactor;
        }
        let axes = &mut self.log_axes[self.view as usize];
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Plot")
                .selected_text(self.view.label())
                .show_ui(ui, |ui| {
                    for view in View::ALL.into_iter().filter(|v| v.offered(detailed)) {
                        ui.selectable_value(&mut self.view, view, view.label());
                    }
                });
            ui.checkbox(&mut axes.x, "log q");
            ui.checkbox(&mut axes.y, "log y");
        });
        let axes = self.log_axes[self.view as usize];
        let curve = run.curve();
        let all = self.view.curves(&curve, detailed);
        let dark = ui.visuals().dark_mode;
        // One id per view and scale, so each fits its own bounds when shown.
        // Also per series set, so S_eff appearing doesn't inherit stale bounds.
        Plot::new(("manybody_plot", self.view.label(), axes.x, axes.y, detailed))
            .legend(egui_plot::Legend::default())
            .x_axis_label(log_label("q (Å⁻¹)", axes.x))
            .y_axis_label(log_label(self.view.axis_label(), axes.y))
            .show(ui, |plot| {
                for s in &all {
                    draw_series(plot, &curve.q, s, axes, dark);
                }
            });
    }

    /// Follows the recent energy range until the user drags or zooms;
    /// double-click resumes following.
    fn ui_energy_trace(&mut self, ui: &mut egui::Ui, color: egui::Color32) {
        let follow = self.follow_energy;
        let trace = &self.energy_trace;
        let response = Plot::new("energy_trace")
            .legend(egui_plot::Legend::default())
            .x_axis_label("Sweeps")
            .y_axis_label("U / N (kT)")
            .show(ui, |plot| {
                marked_line(plot, "U/N", trace.clone(), color);
                if let Some((lo, hi)) = recent_range(trace).filter(|_| follow) {
                    plot.set_plot_bounds_y(lo..=hi);
                }
            })
            .response;
        let zoomed = response.hovered()
            && ui.input(|i| i.zoom_delta() != 1.0 || i.smooth_scroll_delta != egui::Vec2::ZERO);
        if response.double_clicked() {
            self.follow_energy = true;
        } else if response.dragged() || zoomed {
            self.follow_energy = false;
        }
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
        let (translation, rotation) = stats.steps;
        let tuning = if progress.stage == Stage::Equilibrating && info.tune_steps {
            " (tuning)"
        } else {
            ""
        };
        ui.label(format!(
            "Steps{tuning}: translate ±{translation:.2} Å, rotate ±{rotation:.3} rad"
        ));
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

/// egui_plot has no log axes, so log scales plot log₁₀ of the data;
/// non-positive values have no logarithm and are dropped.
fn scaled(v: f64, log: bool) -> Option<f64> {
    match log {
        false => Some(v),
        true => (v > 0.0).then(|| v.log10()),
    }
}

fn point(q: f64, v: f64, axes: LogAxes) -> Option<[f64; 2]> {
    Some([scaled(q, axes.x)?, scaled(v, axes.y)?])
}

fn to_points(q: &[f64], values: &[f64], axes: LogAxes) -> Vec<[f64; 2]> {
    q.iter()
        .zip(values)
        .filter_map(|(&q, &v)| point(q, v, axes))
        .collect()
}

fn log_label(label: &str, log: bool) -> String {
    if log {
        format!("log₁₀ {label}")
    } else {
        label.to_string()
    }
}

/// The Twobody tab's PMF blue, so both tabs read alike.
fn curve_color(dark_mode: bool) -> egui::Color32 {
    if dark_mode {
        egui::Color32::from_rgb(100, 150, 255)
    } else {
        egui::Color32::from_rgb(30, 60, 180)
    }
}

fn style_color(style: Style, dark_mode: bool) -> egui::Color32 {
    let rgb = |dark: (u8, u8, u8), light: (u8, u8, u8)| {
        let (r, g, b) = if dark_mode { dark } else { light };
        egui::Color32::from_rgb(r, g, b)
    };
    match style {
        Style::Primary => curve_color(dark_mode),
        Style::Secondary => rgb((90, 210, 140), (0, 130, 70)),
        Style::Reference => rgb((255, 170, 80), (200, 90, 0)),
    }
}

fn draw_series(
    plot: &mut egui_plot::PlotUi,
    q: &[f64],
    s: &Series,
    axes: LogAxes,
    dark_mode: bool,
) {
    let color = style_color(s.style, dark_mode);
    let points = to_points(q, s.values, axes);
    if let Style::Reference = s.style {
        plot.line(
            Line::new(s.name, PlotPoints::from(points))
                .color(color)
                .width(2.0_f32)
                .style(egui_plot::LineStyle::dashed_dense()),
        );
        return;
    }
    for bar in s
        .errors
        .map(|e| error_bars(q, s.values, e, axes))
        .into_iter()
        .flatten()
    {
        plot.line(Line::new(s.name, PlotPoints::from(bar.to_vec())).color(color));
    }
    marked_line(plot, s.name, points, color);
}

/// U-range over the last half of the sweeps, where the run still drifts
/// slowly; the early drop would otherwise flatten it.
fn recent_range(trace: &[[f64; 2]]) -> Option<(f64, f64)> {
    let half = trace.last()?[0] / 2.0;
    let (lo, hi) = trace
        .iter()
        .filter(|[sweep, _]| *sweep >= half)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), [_, u]| {
            (lo.min(*u), hi.max(*u))
        });
    // A flat trace still needs a visible span.
    let pad = 0.05 * (hi - lo).max(1e-3 * lo.abs().max(1.0));
    Some((lo - pad, hi + pad))
}

/// Twobody style: a line through symbols marking the sampled points.
fn marked_line(
    plot: &mut egui_plot::PlotUi,
    name: &str,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
) {
    plot.line(
        Line::new(name, PlotPoints::from(points.clone()))
            .color(color)
            .width(3.0_f32),
    );
    // Same name, so one legend entry toggles line and markers together.
    plot.points(
        Points::new(name, PlotPoints::from(points))
            .color(color)
            .shape(MarkerShape::Circle)
            .radius(4.0_f32),
    );
}

/// Vertical ±σ bars, one per q; a bar whose end has no logarithm is left out.
fn error_bars(q: &[f64], values: &[f64], errors: &[f64], axes: LogAxes) -> Vec<[[f64; 2]; 2]> {
    (0..q.len())
        .filter_map(|k| {
            let end = |sign: f64| point(q[k], values[k] + sign * errors[k], axes);
            Some([end(-1.0)?, end(1.0)?])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A large initial drop must not set the scale: only the last half of
    /// the sweeps counts, padded by 5% of its span.
    #[test]
    fn recent_range_ignores_the_initial_drop() {
        let trace = [
            [0.0, 100.0],
            [10.0, 10.0],
            [20.0, -1.0],
            [30.0, -2.0],
            [40.0, -3.0],
        ];
        let (lo, hi) = recent_range(&trace).unwrap();
        // Sweeps ≥ 20: U from −3 to −1, span 2, pad 0.1.
        assert!(
            (lo + 3.1).abs() < 1e-12 && (hi + 0.9).abs() < 1e-12,
            "{lo} {hi}"
        );
    }

    fn curve() -> ScatteringCurve {
        ScatteringCurve {
            q: vec![0.1],
            intensity: vec![1.0],
            form_factor: vec![2.0],
            s_eff: vec![3.0],
            s_eff_error: None,
            s_hsy: vec![4.0],
            s_com: vec![5.0],
            s_com_error: None,
            frames: 1,
        }
    }

    fn names(view: View, detailed: bool) -> Vec<&'static str> {
        let curve = curve();
        view.curves(&curve, detailed)
            .iter()
            .map(|s| s.name)
            .collect()
    }

    #[test]
    fn structure_factor_always_shows_mass_centres_and_reference() {
        assert_eq!(
            names(View::StructureFactor, false),
            ["S_com", "charged sphere (MPB-RMSA)"]
        );
        assert_eq!(
            names(View::StructureFactor, true),
            ["S_com", "S_eff", "charged sphere (MPB-RMSA)"]
        );
    }

    #[test]
    fn bead_views_need_a_detailed_run() {
        let offered = |detailed| {
            View::ALL
                .into_iter()
                .filter(|v| v.offered(detailed))
                .collect::<Vec<_>>()
        };
        assert_eq!(offered(false), [View::StructureFactor]);
        assert_eq!(offered(true), View::ALL);
        assert_eq!(names(View::Intensity, true), ["I(q)"]);
    }

    #[test]
    fn recent_range_of_a_flat_trace_is_not_empty() {
        let (lo, hi) = recent_range(&[[0.0, 5.0], [10.0, 5.0]]).unwrap();
        assert!(lo < 5.0 && hi > 5.0);
        assert!(recent_range(&[]).is_none());
    }
}
