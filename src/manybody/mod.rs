//! Manybody tab: Metropolis MC of N copies of the loaded homo-dimer scan,
//! reporting S_com(q) against charged-sphere theory, optionally bead-level
//! I(q), P(q) and S_eff(q), and run statistics. The left panel holds the
//! inputs and run control; each analysis is a floating window.

mod pane;
mod plots;
mod results;
mod scene;
mod stats;

use crate::conditions::SolutionConditions;
use crate::transport::{Action, Phase, Transport};
use crate::widgets::{grid, placeholder, set_in_onebody};
use crate::windows;
use duello::archive::{CgBody, ScanArchive};
use duello::nbody::{
    box_length, Device, HsyFluid, MsaOverrides, NBodyRun, Progress, RunRecord, RunSpec, Scattering,
    Stage, RESIDUE_FORM_FACTORS,
};
use duello::solution::PARTIAL_SPECIFIC_VOLUME;
use pane::Pane;
pub use results::Opened;
use results::Results;
use std::future::Future;
use std::sync::mpsc;

use crate::budget::{self, Instant};

const TRANSPORT: Transport = Transport {
    run_tip: "Equilibrate, then sample until stopped",
    busy: None,
    stop_tip: "Stop: end the run; its results stay for saving",
};

/// The protein concentration comes from the solution conditions, passed in
/// as `molarity` (mol/L), so Onebody stays its only home.
pub struct ManybodyTab {
    n_bodies: usize,
    copies: usize,
    detailed: bool,
    /// Lattice q-vectors per |q| bin for the bead sums.
    vectors_per_bin: usize,
    /// Starting (or, untuned, fixed) translation (Å) and rotation (rad).
    translation: f64,
    rotation: f64,
    tune_steps: bool,
    /// Off by default: the trajectory fills browser memory and only the
    /// download uses it.
    record_trajectory: bool,
    msa: MsaOverrides,
    /// The scan `msa` was chosen for.
    msa_scan: Option<HsyFluid>,
    /// Why a live run refused the last MSA change.
    msa_error: Option<String>,
    /// Last (sphere, q_min) checked for an MSA solution, and the answer.
    msa_check: Option<(HsyFluid, f64, bool)>,
    run: Option<NBodyRun>,
    /// A run opened from a session; never alongside `run`. The panes read
    /// either through [`Results`].
    opened: Option<Opened>,
    /// GPU setup for a detailed run, which the browser can't block on.
    starting: Option<Starting>,
    /// Bead sums still out on the GPU, so the curve has frames to come.
    unsettled: bool,
    /// Why a detailed run fell back to the CPU.
    gpu_note: Option<String>,
    running: bool,
    /// Sweeps per UI frame, tuned by `budget::retune`.
    sweeps_per_frame: u64,
    progress: Option<Progress>,
    scattering: plots::ScatteringView,
    /// Whether the energy plot tracks the recent range; see `recent_range`.
    follow_energy: bool,
    /// Run and trace epoch last plotted, to notice the trace restarting.
    energy_key: Option<(u64, u64)>,
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

struct Starting {
    spec: RunSpec,
    /// The molecule the bead sums place, to detect a scan swapped meanwhile.
    reference: CgBody,
    scattering: mpsc::Receiver<anyhow::Result<Scattering>>,
}

#[derive(Clone, Copy)]
struct RunInfo {
    n_bodies: usize,
    copies: usize,
    millimolar: f64,
    ionic_millimolar: f64,
    detailed: bool,
    tune_steps: bool,
    record_trajectory: bool,
}

impl RunInfo {
    /// `ionic_strength` (mol/L) is the scan's; the spec doesn't hold it.
    fn new(spec: &RunSpec, ionic_strength: f64) -> Self {
        Self {
            n_bodies: spec.n_bodies,
            copies: spec.copies,
            millimolar: spec.molarity * 1e3,
            ionic_millimolar: ionic_strength * 1e3,
            detailed: !spec.centres_only,
            tune_steps: spec.tune_steps,
            record_trajectory: spec.record_trajectory,
        }
    }
}

impl Default for ManybodyTab {
    fn default() -> Self {
        let defaults = RunSpec::new(1, 1.0);
        Self {
            n_bodies: 200,
            copies: 4,
            detailed: false,
            vectors_per_bin: defaults.vectors_per_bin,
            translation: defaults.translation,
            rotation: defaults.rotation,
            tune_steps: true,
            record_trajectory: false,
            msa: MsaOverrides::default(),
            msa_scan: None,
            msa_error: None,
            msa_check: None,
            run: None,
            opened: None,
            starting: None,
            unsettled: false,
            gpu_note: None,
            running: false,
            sweeps_per_frame: 10,
            progress: None,
            scattering: Default::default(),
            follow_energy: true,
            energy_key: None,
            error: None,
            run_info: None,
            run_id: 0,
            viewer: Default::default(),
            view_copy: 0,
        }
    }
}

impl ManybodyTab {
    /// True while something arrives without user input (sweeps, GPU setup
    /// or late GPU frames), so the app keeps repainting.
    pub fn is_busy(&self) -> bool {
        self.running || self.starting.is_some() || self.unsettled
    }

    /// Advances a running simulation by one frame's budget; called every
    /// frame whichever tab is shown, so switching tabs doesn't pause it.
    /// `archive` completes a GPU setup that finished meanwhile.
    pub fn tick(&mut self, archive: Option<&ScanArchive>) {
        self.poll_start(archive);
        // Every run folds in late GPU frames; a running one steps only once
        // none are out, so they can't pile up faster than the GPU returns them.
        let pending = self.settle();
        if self.running && !pending {
            self.advance();
        }
    }

    /// `archive` is the scan loaded in the Twobody tab, if any.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        archive: Option<&ScanArchive>,
        conditions: &SolutionConditions,
    ) {
        egui::Panel::left("manybody_inputs")
            .min_size(220.0)
            .show(ui, |ui| self.ui_inputs(ui, archive, conditions));
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
        let Some(results) = Results::of(&self.run, &self.opened) else {
            return placeholder(ui, "Start a run to see the energy trace.");
        };
        let trace = results.energy();
        if trace.is_empty() {
            return placeholder(
                ui,
                "Waiting for the first energy check or production frame.",
            );
        }
        // A restarted trace (production) gets a fresh, following view.
        let epoch = self.run.as_ref().map_or(0, |r| r.energy_trace_epoch());
        let key = (self.run_id, epoch);
        if self.energy_key != Some(key) {
            self.energy_key = Some(key);
            self.follow_energy = true;
        }
        plots::energy(ui, trace, &mut self.follow_energy, key);
    }

    fn ui_scattering_pane(&mut self, ui: &mut egui::Ui) {
        let Some(results) = produced(
            ui,
            Results::of(&self.run, &self.opened),
            self.progress,
            "S(q)",
        ) else {
            return;
        };
        let curve = results.curve();
        if curve.frames == 0 {
            return placeholder(ui, "Waiting for the first frames from the GPU.");
        }
        let detailed = self.run_info.is_some_and(|i| i.detailed);
        let id = self.run_id;
        plots::scattering(ui, &curve, &mut self.scattering, detailed, id);
    }

    fn ui_pair_correlation(&mut self, ui: &mut egui::Ui) {
        if let Some(results) = produced(
            ui,
            Results::of(&self.run, &self.opened),
            self.progress,
            "g(r)",
        ) {
            plots::pair_correlation(ui, &results.rdf(), self.run_id);
        }
    }

    /// The live run's, else the opened one's; what Save session writes.
    pub fn results(&self) -> Option<Results<'_>> {
        Results::of(&self.run, &self.opened)
    }

    /// What Save session takes from this tab beyond its inputs, if anything.
    pub fn saved_results(&self) -> Option<&'static str> {
        let trajectory = self.results()?.trajectory().is_some();
        Some(if trajectory {
            "the Manybody run's curves, statistics and trajectory"
        } else {
            "the Manybody run's curves and statistics"
        })
    }

    /// Replaces any run with an opened session's, or none; the inputs are
    /// `restore`'s.
    pub fn show_opened(&mut self, opened: Option<Opened>) {
        (self.run, self.starting) = (None, None);
        (self.running, self.unsettled) = (false, false);
        (self.error, self.gpu_note) = (None, None);
        self.run_id += 1;
        self.viewer.reset();
        self.follow_energy = true;
        let record = opened.as_ref().map(|o| &o.record);
        self.progress = record.map(RunRecord::final_progress);
        self.run_info = record.map(|r| RunInfo::new(&r.spec, r.ionic_strength));
        self.opened = opened;
    }

    /// Runs only while the pane is expanded (see `windows::show`), and
    /// rebuilds the scene only when the shown configuration changes.
    fn ui_simulation(&mut self, ui: &mut egui::Ui) {
        if self.opened.is_some() {
            return placeholder(
                ui,
                "A run opened from file keeps no box; Run starts a new one.",
            );
        }
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
        let Some(results) = Results::of(&self.run, &self.opened) else {
            return placeholder(ui, "Start a run to see its statistics.");
        };
        // A run whose first step failed has no progress; the panel shows why.
        let (Some(progress), Some(info)) = (self.progress, self.run_info) else {
            return;
        };
        let error = &mut self.error;
        egui::ScrollArea::vertical().show(ui, |ui| stats::show(ui, results, progress, info, error));
    }

    fn ui_inputs(
        &mut self,
        ui: &mut egui::Ui,
        archive: Option<&ScanArchive>,
        conditions: &SolutionConditions,
    ) {
        let archive = archive.filter(|a| a.settings.homo_dimer);
        let molarity = conditions.protein_molarity();
        // Controls and errors don't depend on the loaded scan, which may have
        // changed since the run started.
        let blocked = self.start_blocker(archive, molarity);
        let phase = self.phase();
        if let Some(action) = TRANSPORT.ui(ui, phase, blocked, Some(self.can_reset(phase))) {
            self.act(action, archive, molarity);
        }
        let phase = self.phase();
        self.ui_status_line(ui, phase, blocked, molarity);
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::RED, e);
        }
        let Some(archive) = archive else {
            return;
        };
        // Inputs describe the next run, which only Stop then Run starts.
        ui.add_enabled_ui(phase == Phase::Idle, |ui| {
            section(ui, "Monte Carlo");
            self.ui_box(ui, archive, conditions);
            ui.checkbox(&mut self.record_trajectory, "Record trajectory")
                .on_hover_text(
                    "Keep every production frame (up to 256 MB in browser memory) for the \
                     .traj download; `duello nbody-expand` turns it into a bead XTC. Off, \
                     the curves are the same and memory stays small",
                );
            egui::CollapsingHeader::new("Move settings")
                .default_open(false)
                .show(ui, |ui| self.ui_mc_inputs(ui, molarity))
                .header_response
                .on_hover_text("Copies and move steps; the defaults suit most runs");
            self.ui_scattering(ui, molarity);
        });
        // Theory only: editable while a run goes, which takes each change.
        self.ui_msa(ui, archive, molarity);
    }

    fn phase(&self) -> Phase {
        match (self.starting.is_some(), self.live_run().is_some()) {
            (true, _) => Phase::Starting,
            (false, true) if self.running => Phase::Running,
            (false, true) => Phase::Paused,
            (false, false) => Phase::Idle,
        }
    }

    fn can_reset(&self, phase: Phase) -> bool {
        let producing = matches!(
            self.progress.map(|p| p.stage),
            Some(Stage::Production { .. })
        );
        producing && matches!(phase, Phase::Running | Phase::Paused)
    }

    /// Periodic images must stay beyond the table's reach.
    fn box_fits(&self, archive: &ScanArchive, molarity: f64) -> bool {
        self.box_length(molarity) >= 2.0 * archive.table.rmax
    }

    /// Why a new run can't start, if it can't.
    fn start_blocker(&self, archive: Option<&ScanArchive>, molarity: f64) -> Option<&'static str> {
        let Some(archive) = archive else {
            return Some("Run or import a scan of two identical molecules in Twobody first");
        };
        (!self.box_fits(archive, molarity))
            .then_some("The box is smaller than the table: increase N")
    }

    fn act(&mut self, action: Action, archive: Option<&ScanArchive>, molarity: f64) {
        match action {
            Action::Start => {
                if let Some(archive) = archive {
                    self.start(archive, molarity);
                }
            }
            Action::Pause => self.running = false,
            Action::Resume => self.running = true,
            Action::Stop if self.starting.is_some() => self.starting = None,
            Action::Stop => self.stop(),
            Action::ResetStatistics => self.reset_statistics(),
        }
    }

    /// One line under the controls: what the run is doing, or why it can't start.
    fn ui_status_line(
        &self,
        ui: &mut egui::Ui,
        phase: Phase,
        blocked: Option<&str>,
        molarity: f64,
    ) {
        let sweeps = |p: Option<Progress>| p.map_or(0, |p| p.sweep);
        match phase {
            Phase::Starting => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.weak("Setting up GPU…")
                });
            }
            Phase::Running | Phase::Paused => {
                let paused = if self.running { "" } else { "Paused · " };
                let stage = self
                    .progress
                    .map_or("starting", |p| stats::stage_label(p.stage));
                ui.weak(format!(
                    "{paused}{stage} · {} sweeps",
                    sweeps(self.progress)
                ));
            }
            Phase::Idle => {
                ui.weak(self.idle_status(blocked, molarity));
            }
        }
        if let Some(note) = &self.gpu_note {
            ui.colored_label(
                egui::Color32::YELLOW,
                "GPU unavailable; detailed I(q) on CPU",
            )
            .on_hover_text(format!(
                "Bead sums run on the CPU, slowing the page: {note}"
            ));
        }
    }

    fn idle_status(&self, blocked: Option<&str>, molarity: f64) -> String {
        let sweeps = self.progress.map_or(0, |p| p.sweep);
        match blocked {
            Some(why) => why.to_string(),
            None if self.opened.is_some() => format!("Loaded from file · {sweeps} sweeps"),
            None if self.stopped() => {
                format!("Stopped after {sweeps} sweeps; results stay for saving")
            }
            None => format!("Ready: L = {:.0} Å", self.box_length(molarity)),
        }
    }

    /// N and the box it makes at Onebody's concentration, flagged when too
    /// small for the table.
    fn ui_box(
        &mut self,
        ui: &mut egui::Ui,
        archive: &ScanArchive,
        conditions: &SolutionConditions,
    ) {
        let molarity = conditions.protein_molarity();
        grid("manybody_box").show(ui, |ui| {
            ui.label("Molecules, N");
            ui.add(egui::DragValue::new(&mut self.n_bodies).range(2..=2000))
                .on_hover_text("Molecules in the periodic box; more lowers q_min");
            ui.end_row();
        });
        let l = self.box_length(molarity);
        ui.label(format!("Box L = {l:.0} Å"))
            .on_hover_text(set_in_onebody(&format!(
                "Cubic box holding N molecules at {:.2} mM",
                conditions.protein_millimolar
            )));
        if let Some(why) = conditions.stale(&archive.settings) {
            ui.colored_label(egui::Color32::YELLOW, "Scan predates the conditions")
                .on_hover_text(format!(
                    "{why}. The run uses the scan's table and ionic strength; re-scan in \
                     Twobody to use the current conditions"
                ));
        }
        let needed = 2.0 * archive.table.rmax;
        if !self.box_fits(archive, molarity) {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                format!("Box below the table's {needed:.0} Å: increase N"),
            );
        }
    }

    fn ui_scattering(&mut self, ui: &mut egui::Ui, molarity: f64) {
        section(ui, "Scattering");
        let defaults = RunSpec::new(1, 1.0);
        grid("manybody_scattering").show(ui, |ui| {
            ui.label("Detailed I(q)");
            ui.checkbox(&mut self.detailed, "").on_hover_text(format!(
                "Also sum over beads for I(q), P(q) and S_eff: a frame every {} sweeps \
                 instead of {}. The bead sums run on the GPU (CPU if none), so the \
                 curves trail the simulation slightly. S_com and, where it has a \
                 solution, the MSA S(q) are always shown",
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
        let l = self.box_length(molarity);
        ui.label(format!("q_min = {:.4} Å⁻¹", std::f64::consts::TAU / l))
            .on_hover_text("2π/L: the lowest q the periodic box can resolve");
    }

    /// Defaults suit most runs, hence collapsed.
    fn ui_mc_inputs(&mut self, ui: &mut egui::Ui, molarity: f64) {
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
            let half_box = self.box_length(molarity) / 2.0;
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
    fn ui_msa(&mut self, ui: &mut egui::Ui, archive: &ScanArchive, molarity: f64) {
        section(ui, "MSA reference");
        let (derived, before) = match self.live_run().map(NBodyRun::msa) {
            Some(msa) => (msa.derived().clone(), msa.overrides()),
            None => (self.next_run_sphere(archive, molarity), self.msa),
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
                let q_min = std::f64::consts::TAU / self.box_length(molarity);
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
    fn next_run_sphere(&mut self, archive: &ScanArchive, molarity: f64) -> HsyFluid {
        let scan = HsyFluid::for_scan(archive, 1.0);
        if self.msa_scan.as_ref() != Some(&scan) {
            self.msa = MsaOverrides::default();
            self.msa_scan = Some(scan);
        }
        HsyFluid::for_scan(archive, molarity)
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

    fn box_length(&self, molarity: f64) -> f64 {
        box_length(self.n_bodies, molarity)
    }

    /// The run the inputs describe, at protein `molarity` (mol/L).
    pub fn spec(&self, molarity: f64) -> RunSpec {
        let mut spec = RunSpec::new(self.n_bodies, molarity);
        spec.copies = self.copies;
        spec.centres_only = !self.detailed;
        spec.vectors_per_bin = self.vectors_per_bin;
        spec.translation = self.translation;
        spec.rotation = self.rotation;
        spec.tune_steps = self.tune_steps;
        spec.msa = self.msa;
        spec.record_trajectory = self.record_trajectory;
        if !self.detailed {
            // Mass-centre frames cost little next to the sweeps between them.
            spec.sweeps_per_frame = spec.sweeps_per_check;
        }
        spec
    }

    /// Inverse of `spec`, for an opened session; without one the inputs stay.
    /// The concentration is the conditions'; the MSA overrides belong to
    /// `scan`, the opened one, so follow it either way.
    pub fn restore(&mut self, spec: Option<&RunSpec>, scan: Option<&ScanArchive>) {
        self.msa_scan = scan.map(|archive| HsyFluid::for_scan(archive, 1.0));
        let Some(spec) = spec else { return };
        self.n_bodies = spec.n_bodies;
        self.copies = spec.copies;
        self.detailed = !spec.centres_only;
        self.vectors_per_bin = spec.vectors_per_bin;
        self.translation = spec.translation;
        self.rotation = spec.rotation;
        self.tune_steps = spec.tune_steps;
        self.record_trajectory = spec.record_trajectory;
        self.msa = spec.msa;
    }

    /// Detailed runs set up GPU bead sums first; `poll_start` launches them.
    fn start(&mut self, archive: &ScanArchive, molarity: f64) {
        // Panes and Save would otherwise keep an opened run while the GPU sets up.
        (self.opened, self.progress, self.run_info) = (None, None, None);
        let mut spec = self.spec(molarity);
        self.gpu_note = None;
        self.error = None;
        if !self.detailed {
            let run = NBodyRun::start(archive, spec, RESIDUE_FORM_FACTORS);
            return self.launch(archive, run);
        }
        spec.device = Device::Gpu;
        let setup = NBodyRun::scattering(archive, &spec, RESIDUE_FORM_FACTORS);
        self.starting = Some(Starting {
            spec,
            reference: archive.bodies[0].clone(),
            scattering: spawn(setup),
        });
    }

    /// Any setup error falls back to the CPU, which reports it again if it
    /// wasn't the GPU's. That fallback blocks, harmlessly: CPU setup never
    /// awaits.
    fn poll_start(&mut self, archive: Option<&ScanArchive>) {
        let Some(starting) = self.starting.take() else {
            return;
        };
        let built = match starting.scattering.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {
                self.starting = Some(starting);
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => Err(anyhow::anyhow!("GPU setup was lost")),
            Ok(built) => built,
        };
        let Starting {
            mut spec,
            reference,
            ..
        } = starting;
        let Some(archive) = archive.filter(|a| a.bodies[0] == reference) else {
            self.error = Some("the scan changed during GPU setup; press Run again".into());
            return;
        };
        let run = match built {
            Ok(scattering) => NBodyRun::start_with(archive, spec, scattering),
            Err(e) => {
                spec.device = Device::Cpu;
                let run = NBodyRun::start(archive, spec, RESIDUE_FORM_FACTORS);
                self.gpu_note = run.is_ok().then(|| format!("{e:#}"));
                run
            }
        };
        self.launch(archive, run);
    }

    fn launch(&mut self, archive: &ScanArchive, run: anyhow::Result<NBodyRun>) {
        match run {
            Ok(run) => {
                let ionic_strength = archive.settings.ionic_strength;
                self.run_info = Some(RunInfo::new(run.spec(), ionic_strength));
                self.run = Some(run);
                self.opened = None;
                self.running = true;
                self.unsettled = false;
                self.run_id += 1;
                // Another box, perhaps another size: the old view needn't suit it.
                self.viewer.reset();
                self.error = None;
                self.progress = None;
                self.follow_energy = true;
                self.sweeps_per_frame = 10;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn reset_statistics(&mut self) {
        let Some(run) = &mut self.run else { return };
        match run.reset_statistics() {
            Ok(progress) => self.progress = Some(progress),
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// The run has stopped itself; `stop` refreshes what the panel shows.
    fn fail(&mut self, e: anyhow::Error) {
        self.error = Some(format!("{e:#}"));
        self.stop();
    }

    fn stop(&mut self) {
        if let Some(run) = &mut self.run {
            run.stop();
            // A zero-sweep step just reports the stopped state; if even that
            // fails, the last progress stands, marked stopped.
            let last = self.progress.map(|p| Progress {
                stage: Stage::Stopped,
                ..p
            });
            self.progress = run.step(0).ok().or(last);
        }
        // A run that never reported progress has nothing to keep.
        if self.progress.is_none() {
            self.run = None;
        }
        self.running = false;
    }

    /// Folds in GPU frames that came back; true while some are still out.
    fn settle(&mut self) -> bool {
        let Some(run) = &mut self.run else {
            return false;
        };
        match run.settle() {
            Ok(pending) => self.unsettled = pending,
            Err(e) => {
                self.unsettled = false;
                self.fail(e);
            }
        }
        self.unsettled
    }

    /// One budgeted step, then retune the budget.
    fn advance(&mut self) {
        let Some(run) = &mut self.run else { return };
        let was_equilibrating = matches!(
            self.progress.map(|p| p.stage),
            None | Some(Stage::Equilibrating)
        );
        let start = Instant::now();
        let progress = match run.step(self.sweeps_per_frame) {
            Ok(progress) => progress,
            Err(e) => return self.fail(e),
        };
        self.progress = Some(progress);
        // The step may have sent GPU frames; the next tick's settle tells,
        // and repaints until then even if the run is paused or stopped.
        self.unsettled = true;
        if was_equilibrating && progress.stage != Stage::Equilibrating {
            // Production steps cost far more per sweep: start the budget over.
            self.sweeps_per_frame = 1;
            return;
        }
        self.sweeps_per_frame = budget::retune(self.sweeps_per_frame, start);
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
            "Screening length κ⁻¹ of the Yukawa repulsion. Default: from the scan's \
             ionic strength, the same κ the MC table screens with",
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

/// `results` once they have production frames; until then a placeholder
/// saying what `what` waits for. Free, so the caller keeps its fields.
fn produced<'a>(
    ui: &mut egui::Ui,
    results: Option<Results<'a>>,
    progress: Option<Progress>,
    what: &str,
) -> Option<Results<'a>> {
    let Some(results) = results else {
        placeholder(ui, &format!("Start a run to see {what}."));
        return None;
    };
    if progress.is_none_or(|p| p.frames == 0) {
        placeholder(ui, "No production frames yet.");
        return None;
    }
    Some(results)
}

/// A Twobody-style heading: rule, bold title.
fn section(ui: &mut egui::Ui, title: &str) {
    crate::widgets::section(ui);
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
/// Runs `future` without blocking the browser's event loop, which GPU
/// readbacks need; natively it simply completes here.
fn spawn<T: 'static>(future: impl Future<Output = T> + 'static) -> mpsc::Receiver<T> {
    let (tx, rx) = mpsc::channel();
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        let _ = tx.send(future.await);
    });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = tx.send(pollster::block_on(future));
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A saved session's inputs must come back as the same run.
    #[test]
    fn restored_inputs_give_back_their_spec() {
        let mut detailed = RunSpec::new(300, 0.004);
        detailed.copies = 6;
        detailed.centres_only = false;
        detailed.vectors_per_bin = 32;
        detailed.translation = 2.0;
        detailed.rotation = 0.3;
        detailed.tune_steps = false;
        detailed.msa.diameter = Some(30.0);
        detailed.record_trajectory = true;
        // Mass-centre runs take a frame at every check.
        let centres = RunSpec {
            centres_only: true,
            sweeps_per_frame: 10,
            record_trajectory: false,
            ..detailed.clone()
        };
        for spec in [detailed, centres] {
            let mut tab = ManybodyTab::default();
            tab.restore(Some(&spec), None);
            assert_eq!(tab.spec(0.004), spec);
        }
    }

    /// A saved run opens as a finished result: idle, so Run starts afresh,
    /// with the panes on the saved curves.
    #[test]
    fn an_opened_run_is_idle_and_shows_what_was_saved() {
        let archive = crate::test_archive::archive();
        let molarity = 0.01;
        let mut live = ManybodyTab {
            n_bodies: 8,
            copies: 1,
            ..Default::default()
        };
        live.start(&archive, molarity);
        for _ in 0..20 {
            live.tick(Some(&archive));
        }
        live.stop();
        let sweeps = live.progress.unwrap().sweep;
        let saved = live.results().unwrap().record().into_owned();
        assert!(sweeps > 0 && saved.sweeps == sweeps);

        // No scan: the record alone must describe its run.
        let mut opened = ManybodyTab::default();
        opened.show_opened(Some(Opened {
            record: saved.clone(),
            trajectory: None,
        }));
        assert!(opened.phase() == Phase::Idle);
        let (shown, was) = (opened.results().unwrap(), live.results().unwrap());
        assert_eq!(shown.curve(), was.curve());
        assert_eq!(shown.energy(), was.energy());
        assert_eq!(opened.run_info.unwrap().ionic_millimolar, 100.0);
        let status = opened.idle_status(None, molarity);
        assert_eq!(status, format!("Loaded from file · {sweeps} sweeps"));

        opened.show_opened(None);
        assert!(
            opened.results().is_none(),
            "a session without a run clears it"
        );
    }
}
