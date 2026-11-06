//! Manybody tab: Metropolis MC of N copies of the loaded homo-dimer scan,
//! reporting S_com(q) against charged-sphere theory, optionally bead-level
//! I(q), P(q) and S_eff(q), and run statistics. The left panel holds the
//! inputs and run control; each analysis is a floating window.

mod plots;
mod scene;
mod stats;
mod windows;

use duello::archive::ScanArchive;
use duello::nbody::{
    box_length, HsyFluid, MsaOverrides, NBodyRun, Progress, RunSpec, Stage,
    PARTIAL_SPECIFIC_VOLUME, RESIDUE_FORM_FACTORS,
};
use windows::Pane;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Wall time per UI frame spent stepping the run, so the page stays
/// responsive while the run makes progress.
const FRAME_BUDGET_MS: f64 = 30.0;

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
    msa: MsaOverrides,
    /// The scan `msa` was chosen for.
    msa_scan: Option<HsyFluid>,
    /// Why a live run refused the last MSA change.
    msa_error: Option<String>,
    /// Last (sphere, q_min) checked for an MSA solution, and the answer.
    msa_check: Option<(HsyFluid, f64, bool)>,
    run: Option<NBodyRun>,
    running: bool,
    /// Sweeps per UI frame, tuned to `FRAME_BUDGET_MS`.
    sweeps_per_frame: u64,
    progress: Option<Progress>,
    scattering: plots::ScatteringView,
    /// Whether the energy plot tracks the recent range; see `recent_range`.
    follow_energy: bool,
    error: Option<String>,
    /// What the current run was started with, so editing the inputs or
    /// loading another scan doesn't relabel its results.
    run_info: Option<RunInfo>,
    /// Bumped per run, so plots fit each run's data instead of keeping
    /// the last run's zoom.
    run_id: u64,
    viewer: crate::viewer::Viewer,
    /// Copy shown in the Simulation pane.
    view_copy: usize,
}

#[derive(Clone, Copy)]
struct RunInfo {
    n_bodies: usize,
    copies: usize,
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
            msa: MsaOverrides::default(),
            msa_scan: None,
            msa_error: None,
            msa_check: None,
            run: None,
            running: false,
            sweeps_per_frame: 10,
            progress: None,
            scattering: Default::default(),
            follow_energy: true,
            error: None,
            run_info: None,
            run_id: 0,
            viewer: Default::default(),
            view_copy: 0,
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
            .show(ui, |ui| self.ui_inputs(ui, archive));
        egui::CentralPanel::default().show(ui, |ui| self.ui_windows(ui));
    }

    /// Each pane fetches its data inside its closure, so a collapsed pane
    /// does no work.
    fn ui_windows(&mut self, ui: &mut egui::Ui) {
        for pane in Pane::ALL {
            windows::show(ui, pane, |ui| self.ui_pane(ui, pane));
        }
    }

    fn ui_pane(&mut self, ui: &mut egui::Ui, pane: Pane) {
        match pane {
            Pane::Energy => self.ui_energy(ui),
            Pane::Scattering => self.ui_scattering_pane(ui),
            Pane::PairCorrelation => self.ui_pair_correlation(ui),
            Pane::Stats => self.ui_stats(ui),
            Pane::Simulation => self.ui_simulation(ui),
        }
    }

    fn ui_energy(&mut self, ui: &mut egui::Ui) {
        let trace = self.run.as_ref().map_or(&[][..], NBodyRun::energy_trace);
        if trace.is_empty() {
            return placeholder(ui, "Start a run to see the energy trace.");
        }
        plots::energy(ui, trace, &mut self.follow_energy, self.run_id);
    }

    fn ui_scattering_pane(&mut self, ui: &mut egui::Ui) {
        let Some(run) = self.produced_run(ui, "S(q)") else {
            return;
        };
        let detailed = self.run_info.is_some_and(|i| i.detailed);
        let id = self.run_id;
        plots::scattering(ui, &run.curve(), &mut self.scattering, detailed, id);
    }

    fn ui_pair_correlation(&mut self, ui: &mut egui::Ui) {
        if let Some(run) = self.produced_run(ui, "g(r)") {
            plots::pair_correlation(ui, &run.pair_correlation(), self.run_id);
        }
    }

    /// The run once it has production frames; until then a placeholder
    /// saying what `what` waits for.
    fn produced_run(&self, ui: &mut egui::Ui, what: &str) -> Option<&NBodyRun> {
        let Some(run) = &self.run else {
            placeholder(ui, &format!("Start a run to see {what}."));
            return None;
        };
        if self.progress.is_none_or(|p| p.frames == 0) {
            placeholder(ui, "No production frames yet.");
            return None;
        }
        Some(run)
    }

    /// Runs only while the pane is expanded (see `windows::show`), and
    /// rebuilds the scene only when the shown configuration changes.
    fn ui_simulation(&mut self, ui: &mut egui::Ui) {
        let (Some(run), Some(info)) = (&self.run, self.run_info) else {
            return placeholder(ui, "Start a run to see the box.");
        };
        // A new run may have fewer copies than the one last shown.
        if self.view_copy >= info.copies {
            self.view_copy = 0;
        }
        if info.copies > 1 {
            copy_picker(ui, &mut self.view_copy, info.copies);
        }
        let Ok(frame) = run.frame(self.view_copy) else {
            return;
        };
        let key = (self.run_id, frame.copy, frame.sweep);
        self.viewer
            .ui(ui, key, || scene::scene(&frame, run.reference()));
    }

    fn ui_stats(&mut self, ui: &mut egui::Ui) {
        if self.run.is_none() {
            return placeholder(ui, "Start a run to see its statistics.");
        }
        // A run whose first step failed has no progress; the panel shows why.
        let (Some(run), Some(progress), Some(info)) = (&self.run, self.progress, self.run_info)
        else {
            return;
        };
        let error = &mut self.error;
        egui::ScrollArea::vertical().show(ui, |ui| stats::show(ui, run, progress, info, error));
    }

    fn ui_inputs(&mut self, ui: &mut egui::Ui, archive: Option<&ScanArchive>) {
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
        let mut fits = true;
        ui.add_enabled_ui(idle, |ui| {
            fits = self.ui_solution(ui, archive);
            self.ui_scattering(ui);
            section(ui, "Monte Carlo");
            egui::CollapsingHeader::new("Move settings")
                .default_open(false)
                .show(ui, |ui| self.ui_mc_inputs(ui))
                .header_response
                .on_hover_text("Copies and move steps; the defaults suit most runs");
        });
        // Theory only: editable while a run goes, which takes each change.
        self.ui_msa(ui, archive);
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .add_enabled(idle && fits, egui::Button::new("Run"))
            .on_hover_text("Equilibrate, then sample until stopped")
            .on_disabled_hover_text("Needs a box larger than the table and no run in progress")
            .clicked()
        {
            self.start(archive);
        }
    }

    /// N, concentration and what follows from them; false if the box is
    /// too small for the table.
    fn ui_solution(&mut self, ui: &mut egui::Ui, archive: &ScanArchive) -> bool {
        section(ui, "Solution");
        grid("manybody_solution").show(ui, |ui| {
            ui.label("Molecules, N");
            ui.add(egui::DragValue::new(&mut self.n_bodies).range(2..=2000))
                .on_hover_text("Molecules in the periodic box; more lowers q_min");
            ui.end_row();
            ui.label("Concentration");
            ui.add(
                egui::DragValue::new(&mut self.millimolar)
                    .range(0.01..=200.0)
                    .speed(0.1)
                    .suffix(" mM"),
            )
            .on_hover_text("Protein molarity; with N it sets the box size");
            ui.end_row();
        });
        let l = self.box_length();
        ui.label(format!("Box L = {l:.0} Å"))
            .on_hover_text("Cubic box holding N molecules at this concentration");
        // Debye length for 1:1 salt in water at 25 °C.
        let debye = 3.04 / archive.settings.molarity.sqrt();
        ui.label(format!(
            "Salt {:.0} mM, Debye length ≈ {debye:.0} Å",
            archive.settings.molarity * 1e3
        ))
        .on_hover_text("Set by the Twobody scan, whose table holds the screened energies");
        let needed = 2.0 * archive.table.rmax;
        let fits = l >= needed;
        if !fits {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                format!("Box below the table's {needed:.0} Å: increase N"),
            );
        }
        fits
    }

    fn ui_scattering(&mut self, ui: &mut egui::Ui) {
        section(ui, "Scattering");
        let defaults = RunSpec::new(1, 1.0);
        grid("manybody_scattering").show(ui, |ui| {
            ui.label("Detailed I(q)");
            ui.checkbox(&mut self.detailed, "").on_hover_text(format!(
                "Also sum over beads for I(q), P(q) and S_eff: a frame every {} sweeps \
                 instead of {}. S_com and, where it has a solution, the MSA S(q) are \
                 always shown",
                defaults.sweeps_per_frame, defaults.sweeps_per_check
            ));
            ui.end_row();
            ui.label("q-vectors per bin");
            ui.add_enabled(
                self.detailed,
                egui::DragValue::new(&mut self.vectors_per_bin).range(1..=500),
            )
            .on_hover_text(
                "Lattice q-vectors per 2π/L bin for the bead sums: cost grows with this; \
                 more gives smoother I(q) and S_eff",
            )
            .on_disabled_hover_text("Only used with Detailed I(q)");
            ui.end_row();
        });
        let l = self.box_length();
        ui.label(format!("q_min = {:.4} Å⁻¹", std::f64::consts::TAU / l))
            .on_hover_text("2π/L: the lowest q the periodic box can resolve");
    }

    /// Defaults suit most runs, hence collapsed.
    fn ui_mc_inputs(&mut self, ui: &mut egui::Ui) {
        grid("manybody_mc").show(ui, |ui| {
            ui.label("Independent copies");
            ui.add(egui::DragValue::new(&mut self.copies).range(1..=16))
                .on_hover_text("Error bars come from the spread between copies");
            ui.end_row();
            ui.label("Tune move steps");
            ui.checkbox(&mut self.tune_steps, "").on_hover_text(
                "During equilibration, find the steps that move molecules furthest per \
                 attempted move; frozen for production",
            );
            ui.end_row();
            let start = if self.tune_steps { " (start)" } else { "" };
            ui.label(format!("Translation step{start}"));
            // Steps beyond L/2 add nothing (and are clamped).
            let half_box = self.box_length() / 2.0;
            ui.add(
                egui::DragValue::new(&mut self.translation)
                    .range(0.01..=half_box)
                    .speed(0.1)
                    .suffix(" Å"),
            )
            .on_hover_text("Largest displacement per axis, up to L/2");
            ui.end_row();
            ui.label(format!("Rotation step{start}"));
            ui.add(
                egui::DragValue::new(&mut self.rotation)
                    .range(0.0..=std::f64::consts::PI)
                    .speed(0.01)
                    .suffix(" rad"),
            )
            .on_hover_text("Largest rotation angle, up to π");
            ui.end_row();
        });
    }

    /// A live (running or paused) run shows and takes edits to its own
    /// sphere; otherwise the section previews the next run from the inputs.
    fn ui_msa(&mut self, ui: &mut egui::Ui, archive: &ScanArchive) {
        section(ui, "MSA reference");
        let (derived, before) = match self.live_run().map(NBodyRun::msa) {
            Some(msa) => (msa.derived().clone(), msa.overrides()),
            None => (self.next_run_sphere(archive), self.msa),
        };
        let mut overrides = before;
        msa_fields(ui, &mut overrides, &derived);
        if overrides != before {
            self.choose_msa(overrides);
        }
        let (sphere, solved) = match self.live_run().map(NBodyRun::msa) {
            Some(msa) => (msa.sphere().clone(), msa.solved()),
            None => {
                let sphere = derived.overridden(&self.msa);
                let q_min = std::f64::consts::TAU / self.box_length();
                (sphere.clone(), self.msa_solvable(sphere, q_min))
            }
        };
        ui.label(format!(
            "Volume fraction φ = {:.4}",
            sphere.volume_fraction()
        ))
        .on_hover_text("Number density × πσ³/6");
        ui.label(format!("Charge Z = {:+.1} e", sphere.charge))
            .on_hover_text("Net charge of the scanned molecule");
        if !solved {
            ui.colored_label(egui::Color32::YELLOW, "No MSA solution: curve not shown")
                .on_hover_text("Too dense or too strongly repulsive for the rescaling");
        }
        if let Some(e) = &self.msa_error {
            ui.colored_label(egui::Color32::RED, e);
        }
    }

    /// Not stopped, so its theory curve is still the one being watched.
    fn live_run(&self) -> Option<&NBodyRun> {
        self.run.as_ref().filter(|_| !self.stopped())
    }

    fn stopped(&self) -> bool {
        self.progress.is_some_and(|p| p.stage == Stage::Stopped)
    }

    /// Overrides chosen for one molecule or solution mean nothing for
    /// another scan, so a new scan clears them; the 1 M sphere tags the scan
    /// without the concentration.
    fn next_run_sphere(&mut self, archive: &ScanArchive) -> HsyFluid {
        let scan = HsyFluid::for_scan(archive, 1.0);
        if self.msa_scan.as_ref() != Some(&scan) {
            self.msa = MsaOverrides::default();
            self.msa_scan = Some(scan);
        }
        HsyFluid::for_scan(archive, self.millimolar * 1e-3)
    }

    /// A live run takes the change or refuses it, keeping its reference; the
    /// next run starts from the last accepted choice either way.
    fn choose_msa(&mut self, overrides: MsaOverrides) {
        let stopped = self.stopped();
        let live = self.run.as_mut().filter(|_| !stopped);
        self.msa_error = match live.map(|run| run.set_msa(overrides)) {
            Some(Err(e)) => Some(format!("{e:#}")),
            _ => None,
        };
        if self.msa_error.is_none() {
            self.msa = overrides;
        }
    }

    /// The panel redraws every frame; the solve is only redone when the
    /// sphere or q_min changes.
    fn msa_solvable(&mut self, fluid: HsyFluid, q_min: f64) -> bool {
        match &self.msa_check {
            Some((checked, q, solvable)) if *checked == fluid && *q == q_min => *solvable,
            _ => {
                let solvable = fluid.structure_factor(&[q_min]).is_some();
                self.msa_check = Some((fluid, q_min, solvable));
                solvable
            }
        }
    }

    fn box_length(&self) -> f64 {
        box_length(self.n_bodies, self.millimolar * 1e-3)
    }

    fn start(&mut self, archive: &ScanArchive) {
        let mut spec = RunSpec::new(self.n_bodies, self.millimolar * 1e-3);
        spec.copies = self.copies;
        spec.centres_only = !self.detailed;
        spec.vectors_per_bin = self.vectors_per_bin;
        spec.translation = self.translation;
        spec.rotation = self.rotation;
        spec.tune_steps = self.tune_steps;
        spec.msa = self.msa;
        if !self.detailed {
            // Mass-centre frames cost little next to the sweeps between them.
            spec.sweeps_per_frame = spec.sweeps_per_check;
        }
        match NBodyRun::start(archive, spec, RESIDUE_FORM_FACTORS) {
            Ok(run) => {
                self.run = Some(run);
                self.run_info = Some(RunInfo {
                    n_bodies: self.n_bodies,
                    copies: self.copies,
                    millimolar: self.millimolar,
                    salt_millimolar: archive.settings.molarity * 1e3,
                    detailed: self.detailed,
                    tune_steps: self.tune_steps,
                });
                self.running = true;
                self.run_id += 1;
                self.error = None;
                self.progress = None;
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
            let hint = if self.running {
                "Hold the run; resume where it left off"
            } else {
                "Continue the held run"
            };
            if ui.button(pause).on_hover_text(hint).clicked() {
                self.running = !self.running;
            }
            if ui
                .button("Stop")
                .on_hover_text("End the run; its results stay for saving")
                .clicked()
            {
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
}

fn msa_fields(ui: &mut egui::Ui, overrides: &mut MsaOverrides, derived: &HsyFluid) {
    grid("manybody_msa").show(ui, |ui| {
        overridable(
            ui,
            "Diameter σ",
            &mut overrides.diameter,
            derived.diameter,
            &format!(
                "Hard-sphere diameter. Default: a sphere of the molecule's volume, \
                 molar mass × {PARTIAL_SPECIFIC_VOLUME} cm³/g (a typical protein \
                 partial specific volume)"
            ),
        );
        overridable(
            ui,
            "Debye length",
            &mut overrides.debye_length,
            derived.debye_length,
            "Screening length κ⁻¹ of the Yukawa repulsion. Default: the scan's salt \
             plus the protein counterions, as in the MPB-RMSA of Kaieda et al. 2014 \
             (the MC table screens with salt only)",
        );
    });
}

/// A length that shows its derived default until edited; the reset button
/// returns to tracking the default.
fn overridable(
    ui: &mut egui::Ui,
    label: &str,
    choice: &mut Option<f64>,
    default: f64,
    hover: &str,
) {
    ui.label(label).on_hover_text(hover);
    ui.horizontal(|ui| {
        let mut value = choice.unwrap_or(default);
        let edited = ui
            .add(
                egui::DragValue::new(&mut value)
                    .range(0.1..=1e4)
                    // Else a default outside the range is clamped, which
                    // counts as an edit and stores an override.
                    .clamp_existing_to_range(false)
                    .speed(0.1)
                    .max_decimals(1)
                    .suffix(" Å"),
            )
            .on_hover_text(hover)
            .changed();
        if edited {
            *choice = Some(value);
        }
        let reset = ui
            .add_enabled(choice.is_some(), egui::Button::new("↺").small())
            .on_hover_text(format!("Back to the default, {default:.1} Å"))
            .on_disabled_hover_text("Showing the default");
        if reset.clicked() {
            *choice = None;
        }
    });
    ui.end_row();
}

/// A Twobody-style heading: rule, bold title.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(2.0);
    ui.strong(title);
    ui.add_space(2.0);
}

fn copy_picker(ui: &mut egui::Ui, copy: &mut usize, copies: usize) {
    egui::ComboBox::from_label("Copy")
        .selected_text(format!("{}", *copy + 1))
        .show_ui(ui, |ui| {
            for c in 0..copies {
                ui.selectable_value(copy, c, format!("{}", c + 1));
            }
        })
        .response
        .on_hover_text("Which independent copy of the system to show");
}

/// What a pane shows until it has data.
fn placeholder(ui: &mut egui::Ui, text: &str) {
    ui.centered_and_justified(|ui| ui.weak(text));
}

fn grid(id: &str) -> egui::Grid {
    egui::Grid::new(id).num_columns(2).spacing([8.0, 4.0])
}
