mod curves;
mod job;
mod pane;
mod poses;
mod reference;
mod solution_scan;
mod summary;
mod sweep;
mod thermo;
use crate::compute::{ComputeResult, ComputeStatus};
use crate::conditions::SolutionConditions;
use crate::files::save_file;
use crate::forcefield::ForceField;
use crate::inputs::ScanInputs;
use crate::structures::{StructureId, Structures};
use crate::transport::{Action, Phase, Transport};
use crate::widgets::{pick, placeholder, row, section};
use crate::windows;
use curves::{Curves, Shown};
use duello::archive::{ScanArchive, StructureFile};
use duello::web_api::ScanBackend;
use duello::web_api::{ScanProgress, ScanSettings};
use duello::Subdivision;
use job::{JobEvent, Retired, ScanJob};
use pane::Pane;
use poses::PosesView;
pub use reference::Normaliser;
use solution_scan::{SolutionScan, Start, SweepAction};
use summary::Facts;

/// Why Run is disabled, kept on the status line while it applies.
const NO_GPU: &str =
    "No GPU/WebGPU adapter found. A WebGPU-capable browser is required (Chrome 113+, Edge 113+).";

/// Typical protein separations; typed values may go past either end. Not
/// from 0, which a drag would otherwise reach: full overlap.
const R_SLIDER: std::ops::RangeInclusive<f64> = 5.0..=150.0;

fn backend_label(use_grid: bool) -> &'static str {
    if use_grid {
        "Grid (fast)"
    } else {
        "Exact pairs"
    }
}

fn subdivision_label(scheme: Subdivision) -> &'static str {
    match scheme {
        Subdivision::Geodesic => "Geodesic",
        Subdivision::Lattice => "Lattice",
    }
}

/// A scan can't be held, only stopped, which drops it.
const TRANSPORT: Transport = Transport {
    run_tip: "Tabulate the pair energy over separations and orientations",
    busy: Some("Scanning…"),
    stop_tip: "Stop: abandon the scan; its partial curves are dropped",
};

/// Two-body scan of a pair picked from Onebody's structures: inputs, the
/// scan's lifecycle, exports and the PMF plot.
pub struct TwobodyTab {
    inputs: ScanInputs,
    /// Molecules 1 and 2; the same entry twice is a homo-dimer.
    selected: [Option<StructureId>; 2],
    /// The pair the shown result was scanned from, with the chains each kept,
    /// for export; None when unknown, e.g. opened from a CLI scan.
    scanned: Option<[(StructureId, Vec<String>); 2]>,
    /// The last scan asked for the grid but ran on exact pairs.
    fell_back: bool,
    status: ComputeStatus,
    result: Option<ComputeResult>,
    /// Streamed while a scan runs, then replaced by the result's.
    curves: Option<Curves>,
    shown: Shown,
    poses: PosesView,
    error: Option<String>,
    gpu_available: bool,
    /// The running scan, if any.
    job: Option<ScanJob>,
    solution: SolutionScan,
    /// Clicked inside the sweep's pane, which lacks the inputs to act on it.
    sweep_action: Option<SweepAction>,
    /// Stopped scans still finishing an R-bin on the GPU.
    retired: Retired,
}

impl TwobodyTab {
    pub fn new() -> Self {
        // On WASM, we can't synchronously check GPU availability (pollster::block_on panics).
        // Assume available; errors will surface when the scan actually starts.
        #[cfg(target_arch = "wasm32")]
        let gpu_available = true;
        #[cfg(not(target_arch = "wasm32"))]
        let gpu_available = duello::backend::GpuBackend::is_available();
        if !gpu_available {
            log::warn!("No GPU adapter found. Scan requires WebGPU support.");
        }
        Self {
            inputs: ScanInputs::default(),
            selected: [None; 2],
            scanned: None,
            fell_back: false,
            status: ComputeStatus::Idle,
            result: None,
            curves: None,
            shown: Shown::default(),
            poses: PosesView::default(),
            error: None,
            gpu_available,
            job: None,
            solution: SolutionScan::default(),
            sweep_action: None,
            retired: Retired::default(),
        }
    }

    /// The finished or imported scan, if any.
    pub fn archive(&self) -> Option<&ComputeResult> {
        self.result.as_ref()
    }

    /// Drains background channels; keeps repainting while a scan runs, since
    /// its progress arrives off the UI thread.
    pub fn poll(&mut self, ctx: &egui::Context) {
        self.poll_job();
        self.solution.poll();
        self.retired.poll();
        if !self.idle() || self.retired.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    /// Shows an opened session's scan, or none, with the panel set to the
    /// tabs' inputs it was saved with.
    pub fn show_opened(&mut self, scan: Option<ScanArchive>, inputs: Option<&ScanSettings>) {
        self.error = None;
        // A sweep belongs to the structures it ran on, which the session replaces.
        self.stop_sweep();
        self.solution = SolutionScan::default();
        self.scanned = None;
        if let Some(inputs) = inputs {
            self.inputs.adopt_settings(inputs);
        }
        self.fell_back = false;
        match scan {
            Some(scan) => {
                // An opened scan has no request of its own: show what made it.
                self.inputs.adopt_backend(scan.backend);
                self.install_result(scan);
            }
            None => {
                self.result = None;
                self.curves = None;
                self.poses.invalidate();
            }
        }
    }

    /// The scan parameters in the panel.
    pub fn inputs(&self) -> &ScanInputs {
        &self.inputs
    }

    /// Shown in the panel, e.g. why an opened session couldn't restore everything.
    pub fn warn(&mut self, message: String) {
        self.error = Some(message);
    }

    /// Picks the restored pair, as the scan shown was made from it with
    /// the chains it was saved with.
    pub fn select_scanned(&mut self, pair: [StructureId; 2], structures: &Structures) {
        self.selected = pair.map(Some);
        let get = |id| Some((id, structures.get(id)?.chains.clone()));
        self.scanned = get(pair[0]).zip(get(pair[1])).map(<[_; 2]>::from);
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        structures: &Structures,
        conditions: &SolutionConditions,
        forcefield: &ForceField,
    ) {
        let is_dark = ui.visuals().dark_mode;
        self.keep_selection_valid(structures);
        egui::Panel::left("inputs").min_size(220.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.ui_left_panel(ui, structures, conditions, forcefield);
            });
        });

        let sweep_blocked = self.sweep_blocker(structures);
        egui::CentralPanel::default().show(ui, |ui| self.ui_windows(ui, is_dark, sweep_blocked));
        match self.sweep_action.take() {
            Some(SweepAction::Start) => self.start_sweep(structures, conditions, forcefield),
            Some(SweepAction::Stop) => self.stop_sweep(),
            Some(SweepAction::Save) => self.download_sweep(),
            None => {}
        }
    }

    /// Each pane fetches its data inside its closure, so a collapsed pane costs nothing.
    fn ui_windows(&mut self, ui: &mut egui::Ui, is_dark: bool, sweep_blocked: Option<&str>) {
        for pane in Pane::ALL {
            windows::show(ui, pane, |ui| {
                self.ui_pane(ui, pane, is_dark, sweep_blocked)
            });
        }
    }

    fn ui_pane(
        &mut self,
        ui: &mut egui::Ui,
        pane: Pane,
        is_dark: bool,
        sweep_blocked: Option<&str>,
    ) {
        match pane {
            Pane::FreeEnergy => match &mut self.curves {
                Some(curves) => curves.ui(ui, &mut self.shown, is_dark),
                None => placeholder(ui, "Load structures in Onebody and run a scan"),
            },
            Pane::Thermodynamics => match &self.result {
                Some(result) => {
                    let summary = Facts::of(result).summary(&self.inputs.normaliser);
                    thermo::thermodynamics(ui, &summary, result.settings.homo_dimer)
                }
                None => placeholder(ui, "Available once the scan finishes"),
            },
            Pane::SolutionScan => {
                let normaliser = &self.inputs.normaliser;
                self.sweep_action = self.solution.ui(ui, normaliser, sweep_blocked);
            }
            // A scan clears the result when it starts, so poses show only once it ends.
            Pane::Poses => self.poses.ui(ui, self.result.as_ref()),
        }
    }

    /// Clears picks whose structure was removed. With neither left, both
    /// take the first structure, so one load is enough for a homo-dimer;
    /// otherwise the user re-picks, so a hetero pair never turns homo.
    fn keep_selection_valid(&mut self, structures: &Structures) {
        let valid = |id: Option<StructureId>| id.filter(|&id| structures.get(id).is_some());
        self.selected = match self.selected.map(valid) {
            [None, None] => [structures.first(); 2],
            kept => kept,
        };
    }

    fn ui_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        structures: &Structures,
        conditions: &SolutionConditions,
        forcefield: &ForceField,
    ) {
        // --- Molecules ---
        ui.strong("Molecules");
        ui.add_space(2.0);
        self.ui_molecules(ui, structures);

        // Force field and conditions are set in Onebody; only a mismatch shows here.
        if let Some(settings) = self.result.as_ref().map(|r| &r.settings) {
            if let Some(why) = forcefield.stale(settings) {
                differs_from_scan(ui, &why, "force field");
            }
            if let Some(why) = conditions.stale(settings) {
                differs_from_scan(ui, &why, "conditions");
            }
            if self.chains_changed(structures) {
                let why = "A scanned structure keeps other chains now";
                differs_from_scan(ui, why, "chains");
            }
        }

        section(ui);

        // --- Scan range ---
        ui.strong("Scan range");
        ui.add_space(2.0);

        egui::Grid::new("scan_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                self.ui_rmin_auto(ui);
                self.ui_r_range(ui);
                let pair = self.pair(structures).map(|[a, b]| [a.1, b.1]);
                let homodimer = pair.is_some_and(|[a, b]| crate::compute::same_molecule(a, b));
                self.inputs.normaliser.ui(ui, homodimer);

                ui.label("\u{0394}R");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.dr)
                        .speed(0.1)
                        .suffix(" \u{00C5}")
                        .range(0.1..=10.0),
                )
                .on_hover_text("Spacing between sampled separations");
                ui.end_row();
            });

        ui.add_space(4.0);

        self.ui_angular_resolution(ui);

        ui.add_space(6.0);

        // --- Advanced (collapsed) ---
        egui::CollapsingHeader::new("Advanced")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("advanced_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        self.ui_energy_backend(ui);
                        self.ui_angular_mesh(ui);
                        ui.end_row();

                        ui.label("Cutoff");
                        ui.add(
                            egui::DragValue::new(&mut self.inputs.cutoff)
                                .speed(1.0)
                                .suffix(" \u{00C5}")
                                .range(1.0..=500.0),
                        )
                        .on_hover_text(
                            "Short-range pair potentials are splined out to this distance",
                        );
                        ui.end_row();

                        ui.label("Gradient threshold");
                        ui.add(
                            egui::DragValue::new(&mut self.inputs.gradient_threshold)
                                .speed(0.1)
                                .range(0.0..=10.0),
                        )
                        .on_hover_text(
                            "Boltzmann-weight gradient (1/rad) above which a separation gets \
                            the finer angular mesh; lower refines more",
                        );
                        ui.end_row();

                        ui.label("Spline grid size");
                        ui.add(
                            egui::DragValue::new(&mut self.inputs.grid_size)
                                .speed(10)
                                .range(50..=1000),
                        )
                        .on_hover_text("Points per pair-potential spline");
                        ui.end_row();
                    });
            });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        let blocked = self.start_blocker(structures);
        match TRANSPORT.ui(ui, self.phase(), blocked, None) {
            Some(Action::Start) => self.start_scan(structures, conditions, forcefield),
            Some(Action::Stop) => {
                if let Some(job) = self.job.take() {
                    self.retired.retire(job);
                }
                self.abandon_scan();
            }
            _ => {}
        }
        self.ui_status(ui, blocked);
    }

    /// Why the main scan can't start: one GPU job at a time.
    fn start_blocker(&self, structures: &Structures) -> Option<&'static str> {
        if self.solution.running() {
            Some("A Solution scan is running; stop it to scan here")
        } else {
            self.gpu_blocker()
                .or_else(|| self.inputs_blocker(structures))
        }
    }

    fn sweep_blocker(&self, structures: &Structures) -> Option<&'static str> {
        if self.job.is_some() {
            Some("The main scan is running")
        } else {
            self.gpu_blocker()
                .or_else(|| self.inputs_blocker(structures))
        }
    }

    /// A stopped scan holds the GPU until its R-bin ends.
    fn gpu_blocker(&self) -> Option<&'static str> {
        (self.retired.busy()).then_some("A stopped scan is finishing its last separation")
    }

    fn stop_sweep(&mut self) {
        if let Some(job) = self.solution.stop() {
            self.retired.retire(job);
        }
    }

    fn download_sweep(&self) {
        if let Some(csv) = self.solution.csv(&self.inputs.normaliser) {
            self.download_string(&csv, "solution_scan.csv");
        }
    }

    /// Why no scan of the picked pair can start, if none can.
    fn inputs_blocker(&self, structures: &Structures) -> Option<&'static str> {
        if !self.gpu_available {
            Some(NO_GPU)
        } else if self.pair(structures).is_none() {
            Some("Pick both molecules from Onebody's structures")
        } else if !self.inputs.rmin_auto && self.inputs.rmin >= self.inputs.rmax {
            Some("The R range must have a lower end below its upper end")
        } else if self.inputs.dr <= 0.0 {
            Some("\u{0394}R must be positive")
        } else {
            None
        }
    }

    /// Under the run row: the scan's progress, or why it can't start, then any error.
    fn ui_status(&self, ui: &mut egui::Ui, blocked: Option<&str>) {
        match (&self.status, blocked) {
            (&ComputeStatus::Running { current_r, total_r }, _) => {
                let progress = current_r as f32 / total_r as f32;
                ui.add(
                    egui::ProgressBar::new(progress).text(format!("Step {current_r}/{total_r}")),
                )
                .on_hover_text("Separations scanned so far");
            }
            (ComputeStatus::Idle, Some(why)) => {
                ui.weak(why);
            }
            (ComputeStatus::Idle, None) => {}
        }
        if self.fell_back {
            ui.weak("Grid maps didn't fit the GPU; this scan used exact pairs")
                .on_hover_text("Try a coarser grid spacing, or keep exact pairs");
        }
        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
        }
    }

    /// Grid maps or exact pairs, and the map spacing; rows of the Advanced grid.
    fn ui_energy_backend(&mut self, ui: &mut egui::Ui) {
        let inputs = &mut self.inputs;
        row(
            ui,
            "Energy evaluation",
            "Grid: short-range energies read from maps precomputed around one \
            molecule; several times faster, slightly repulsive near contact. Exact \
            pairs: every bead pair summed. Electrostatics are exact either way.",
            |ui| {
                pick(
                    ui,
                    "energy_backend_combo",
                    &mut inputs.use_grid,
                    &[true, false],
                    |g| backend_label(*g),
                    None,
                )
            },
        );
        if inputs.use_grid {
            row(
                ui,
                "Grid spacing",
                "Map node spacing; the bias near contact scales as its square. Each \
                halving costs 8× the map memory, and maps that don't fit the GPU fall \
                back to exact pairs.",
                |ui| {
                    ui.add(
                        egui::DragValue::new(&mut inputs.grid.spacing)
                            .speed(0.05)
                            .suffix(" \u{00C5}")
                            .range(0.25..=2.0),
                    )
                },
            );
        }
    }

    /// How the orientation mesh subdivides; two grid cells.
    fn ui_angular_mesh(&mut self, ui: &mut egui::Ui) {
        ui.label("Angular mesh");
        egui::ComboBox::from_id_salt("subdivision_combo")
            .selected_text(subdivision_label(self.inputs.subdivision))
            .show_ui(ui, |ui| {
                for scheme in [Subdivision::Geodesic, Subdivision::Lattice] {
                    ui.selectable_value(
                        &mut self.inputs.subdivision,
                        scheme,
                        subdivision_label(scheme),
                    );
                }
            })
            .response
            .on_hover_text(
                "How each icosahedron face is subdivided. Geodesic: near-uniform \
                    vertices, found by search. Lattice: a regular grid per face, found in \
                    closed form (faster lookups).",
            );
    }

    /// The orientation mesh's finest level.
    fn ui_angular_resolution(&mut self, ui: &mut egui::Ui) {
        ui.label("Angular resolution");
        let vertices = self.inputs.subdivision.n_vertices(self.inputs.max_ndiv);
        ui.add(egui::Slider::new(&mut self.inputs.max_ndiv, 0..=4))
            .on_hover_text(format!(
                "Finest mesh level, refined only where the energy varies. Level 0 is \
                the 12-vertex icosahedron; this level has {vertices} vertices per sphere."
            ));
    }

    /// Both picked structures, if both are loaded.
    fn pair<'a>(
        &self,
        structures: &'a Structures,
    ) -> Option<[(StructureId, &'a StructureFile); 2]> {
        let get = |id: Option<StructureId>| Some((id?, structures.get(id?)?));
        Some([get(self.selected[0])?, get(self.selected[1])?])
    }

    fn ui_molecules(&mut self, ui: &mut egui::Ui, structures: &Structures) {
        if structures.first().is_none() {
            ui.weak("Load structures in Onebody")
                .on_hover_text("Onebody holds the structures every tab works on");
            return;
        }
        egui::Grid::new("molecules_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for (n, slot) in self.selected.iter_mut().enumerate() {
                    let label = format!("Molecule {}", n + 1);
                    ui.label(&label);
                    molecule_picker(ui, &label, slot, structures);
                    ui.end_row();
                }
            });
    }

    /// The pair and settings as the main Run would take them.
    fn start_sweep(
        &mut self,
        structures: &Structures,
        conditions: &SolutionConditions,
        forcefield: &ForceField,
    ) {
        let Some(picked) = self.pair(structures) else {
            return;
        };
        self.solution.start(Start {
            base: self.inputs.settings(conditions, forcefield),
            grid: self.inputs.grid_choice(),
            pair: picked.map(|(_, file)| file.clone()),
        });
    }

    fn start_scan(
        &mut self,
        structures: &Structures,
        conditions: &SolutionConditions,
        forcefield: &ForceField,
    ) {
        let Some(picked) = self.pair(structures) else {
            return;
        };
        self.fell_back = false;
        self.scanned = Some(picked.map(|(id, file)| (id, file.chains.clone())));
        let pair = picked.map(|(_, file)| file.clone());

        self.error = None;
        self.result = None;
        self.status = ComputeStatus::Running {
            current_r: 0,
            total_r: 1,
        };
        // Empty until the first R-bin arrives; the plot then grows with the scan.
        self.curves = Some(Curves::new());
        let settings = self.inputs.settings(conditions, forcefield);
        self.job = Some(ScanJob::start(settings, self.inputs.grid_choice(), pair));
    }

    /// Drains the running scan's events. Called each frame.
    fn poll_job(&mut self) {
        let Some(job) = &self.job else { return };
        for event in job.poll() {
            match event {
                JobEvent::Progress(p) => self.show_progress(p),
                JobEvent::Done(Ok(result)) => {
                    // Keep the request; the note says this scan used exact pairs.
                    self.fell_back =
                        self.inputs.use_grid && !matches!(result.backend, ScanBackend::Grid(_));
                    self.install_result(*result);
                    self.status = ComputeStatus::Idle;
                    self.job = None;
                }
                JobEvent::Done(Err(e)) => {
                    self.error = Some(format!("{e:#}"));
                    self.abandon_scan();
                }
            }
        }
    }

    fn show_progress(&mut self, p: ScanProgress) {
        self.status = ComputeStatus::Running {
            current_r: p.done,
            total_r: p.total,
        };
        if let (Some(point), Some(curves)) = (p.point, &mut self.curves) {
            curves.push(point);
        }
    }

    /// Ends a scan that won't deliver a result. Drops the partial curve, which would
    /// otherwise look like a finished PMF.
    fn abandon_scan(&mut self) {
        self.status = ComputeStatus::Idle;
        self.job = None;
        self.curves = None;
    }

    /// Auto R_min needs the coarse-grained bodies, so it is settled at scan start.
    fn ui_rmin_auto(&mut self, ui: &mut egui::Ui) {
        let tip = format!(
            "R_min from where every orientation of the two bead sets overlaps (radii × \
             {}, for soft beads), chosen when the scan starts. Turn off to set it by hand; \
             the e^(−w/kT) at σ row checks the choice after the scan.",
            duello::steric::CORE_RADIUS_SCALE
        );
        // From the result: inputs.rmin may have been edited since, with Auto off.
        let note = match &self.result {
            Some(r) => format!("at scan start (last scan: {:.1} \u{00C5})", r.settings.rmin),
            None => "at scan start".into(),
        };
        let inputs = &mut self.inputs;
        row(ui, "R_min", &tip, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut inputs.rmin_auto, "Auto");
                if inputs.rmin_auto {
                    ui.weak(note);
                }
            })
            .response
        });
    }

    /// Both ends by hand, or only R_max while R_min is automatic.
    fn ui_r_range(&mut self, ui: &mut egui::Ui) {
        let tip = "Sampled mass-centre separations. Below the lower end we assume \
                   infinite repulsion in the osmotic second virial coefficient, \
                   B\u{2082}. Beyond the upper end the tail is fitted. Type a value to go \
                   past the slider's ends.";
        let inputs = &mut self.inputs;
        let label = if inputs.rmin_auto { "R_max" } else { "R" };
        row(ui, label, tip, |ui| match inputs.rmin_auto {
            true => ui.add(
                egui::Slider::new(&mut inputs.rmax, R_SLIDER)
                    .clamping(egui::SliderClamping::Never)
                    .suffix(" \u{00C5}"),
            ),
            false => ui.add(
                egui::RangeSlider::new(&mut inputs.rmin, &mut inputs.rmax, R_SLIDER)
                    .min_separation(1.0)
                    .clamping(egui::SliderClamping::Never)
                    .suffix(" \u{00C5}"),
            ),
        });
    }

    /// Shared by finished scans and imports, so both show a result the same way.
    fn install_result(&mut self, result: ComputeResult) {
        if self.inputs.rmin_auto {
            self.inputs.rmin = result.settings.rmin;
        }
        self.curves.get_or_insert_with(Curves::new).finish(&result);
        self.result = Some(result);
        self.poses.invalidate();
    }

    /// Save failures only reach the log, as before the scan export existed.
    fn download_string(&self, content: &str, filename: &str) {
        if let Err(e) = save_file(content.as_bytes(), filename) {
            log::error!("Failed to save {filename}: {e}");
        }
    }

    fn phase(&self) -> Phase {
        match self.status {
            ComputeStatus::Running { .. } => Phase::Running,
            ComputeStatus::Idle => Phase::Idle,
        }
    }

    /// No scan running, so another can be opened.
    pub fn idle(&self) -> bool {
        matches!(self.status, ComputeStatus::Idle) && !self.solution.running()
    }

    pub fn gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// File ▸ Export items: the shown scan's results as plain files.
    pub fn export_menu(&self, ui: &mut egui::Ui) {
        self.export_sweep(ui);
        let Some(result) = &self.result else { return };
        if ui
            .button("PMF (.csv)")
            .on_hover_text("Free energy and mean energy against separation")
            .clicked()
        {
            self.download_csv();
        }
        for (n, body) in result.bodies.iter().enumerate() {
            if ui
                .button(format!("Molecule {} (.xyz)", n + 1))
                .on_hover_text(format!("Coarse-grained beads of molecule {}", n + 1))
                .clicked()
            {
                self.download_string(&body.to_xyz(), &format!("mol{}_cg.xyz", n + 1));
            }
        }
        if ui
            .button("Topology (.yaml)")
            .on_hover_text("Bead kinds and pair potentials used by the scan")
            .clicked()
        {
            self.download_string(&result.topology_yaml, "topology.yaml");
        }
    }

    fn export_sweep(&self, ui: &mut egui::Ui) {
        if !self.solution.has_points() {
            return;
        }
        if ui
            .button("Solution scan (.csv)")
            .on_hover_text(solution_scan::CSV_TIP)
            .clicked()
        {
            self.download_sweep();
        }
    }

    /// Whether a scanned structure, still loaded, now keeps other chains.
    fn chains_changed(&self, structures: &Structures) -> bool {
        (self.scanned.iter().flatten()).any(|s| keeps_scanned(structures, s) == Some(false))
    }

    /// The pair the shown result was scanned from, if known and both still
    /// keep the chains scanned; a running scan already names its pair, but
    /// shows no result yet.
    pub fn scanned(&self, structures: &Structures) -> Option<[StructureId; 2]> {
        let scanned = self.result.as_ref().and(self.scanned.as_ref())?;
        let unchanged = scanned
            .iter()
            .all(|s| keeps_scanned(structures, s) == Some(true));
        unchanged.then(|| scanned.each_ref().map(|(id, _)| *id))
    }

    fn download_csv(&self) {
        let Some(result) = &self.result else { return };
        let csv = duello::report::pmf_csv(&result.points);
        self.download_string(&csv, "pmf.csv");
    }
}

/// Whether a scanned structure still keeps the chains it was scanned with;
/// None once removed.
fn keeps_scanned(
    structures: &Structures,
    (id, chains): &(StructureId, Vec<String>),
) -> Option<bool> {
    structures.get(*id).map(|f| f.chains == *chains)
}

/// Warns that Onebody's `what` no longer matches the scan on screen.
fn differs_from_scan(ui: &mut egui::Ui, why: &str, what: &str) {
    ui.colored_label(egui::Color32::YELLOW, "Differs from the shown scan")
        .on_hover_text(format!("{why}. Run a new scan to use the {what} above"));
}

/// Combo over the loaded structures for one molecule slot.
fn molecule_picker(
    ui: &mut egui::Ui,
    label: &str,
    slot: &mut Option<StructureId>,
    structures: &Structures,
) {
    let selected = slot.and_then(|id| structures.label(id));
    egui::ComboBox::from_id_salt(label)
        .selected_text(selected.unwrap_or_else(|| "—".into()))
        .show_ui(ui, |ui| {
            for (id, _) in structures.list() {
                let text = structures.label(id).unwrap_or_default();
                ui.selectable_value(slot, Some(id), text);
            }
        })
        .response
        .on_hover_text(
            "A structure loaded in Onebody. Picking the same one for both molecules \
            scans a homo-dimer, which Manybody needs.",
        );
}
