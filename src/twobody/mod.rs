mod curves;
mod pane;
mod poses;
mod thermo;
use crate::compute::{ComputeResult, ComputeStatus};
use crate::conditions::SolutionConditions;
use crate::files::save_file;
use crate::forcefield::ForceField;
use crate::inputs::ScanInputs;
use crate::structures::{StructureId, Structures};
use crate::transport::{Action, Phase, Transport};
use crate::widgets::{placeholder, section};
use crate::windows;
use curves::{Curves, Shown};
use duello::archive::{ScanArchive, StructureFile};
use duello::web_api::{ScanProgress, ScanSettings};
use duello::Subdivision;
use pane::Pane;
use poses::{poses_shown, PosesView};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

/// Why Run is disabled, kept on the status line while it applies.
const NO_GPU: &str =
    "No GPU/WebGPU adapter found. A WebGPU-capable browser is required (Chrome 113+, Edge 113+).";

/// Typical protein separations; typed values may go past either end. Not
/// from 0, which a drag would otherwise reach: full overlap.
const R_SLIDER: std::ops::RangeInclusive<f64> = 5.0..=150.0;

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
    /// The pair the shown result was scanned from, for export; None when
    /// unknown, e.g. opened from a CLI scan.
    scanned: Option<[StructureId; 2]>,
    status: ComputeStatus,
    result: Option<ComputeResult>,
    /// Streamed while a scan runs, then replaced by the result's.
    curves: Option<Curves>,
    shown: Shown,
    poses: PosesView,
    error: Option<String>,
    gpu_available: bool,
    /// Channel for receiving results from background scan thread.
    result_rx: Option<mpsc::Receiver<anyhow::Result<ComputeResult>>>,
    /// Progress and PMF points, one message per finished R-bin.
    progress_rx: Option<mpsc::Receiver<ScanProgress>>,
    /// Set by Stop. One per scan, so a new scan can't un-cancel an abandoned one
    /// that has yet to reach its next R-bin.
    cancel_flag: Arc<AtomicBool>,
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
            status: ComputeStatus::Idle,
            result: None,
            curves: None,
            shown: Shown::default(),
            poses: PosesView::default(),
            error: None,
            gpu_available,
            result_rx: None,
            progress_rx: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The finished or imported scan, if any.
    pub fn archive(&self) -> Option<&ComputeResult> {
        self.result.as_ref()
    }

    /// Drains background channels; keeps repainting while a scan runs, since
    /// its progress arrives off the UI thread.
    pub fn poll(&mut self, ctx: &egui::Context) {
        self.poll_result();
        self.poll_progress();
        if matches!(self.status, ComputeStatus::Running { .. }) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    /// Shows an opened session's scan, or none, with the panel set to the
    /// tabs' inputs it was saved with.
    pub fn show_opened(&mut self, scan: Option<ScanArchive>, inputs: Option<&ScanSettings>) {
        self.error = None;
        self.scanned = None;
        if let Some(inputs) = inputs {
            self.inputs = ScanInputs::from_settings(inputs);
        }
        match scan {
            Some(scan) => self.install_result(scan),
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

    /// Picks the restored pair, as the scan shown was made from it.
    pub fn select_scanned(&mut self, pair: [StructureId; 2]) {
        self.selected = pair.map(Some);
        self.scanned = Some(pair);
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

        egui::CentralPanel::default().show(ui, |ui| self.ui_windows(ui, is_dark));
    }

    /// Each pane fetches its data inside its closure, so a collapsed pane costs nothing.
    fn ui_windows(&mut self, ui: &mut egui::Ui, is_dark: bool) {
        for pane in Pane::ALL {
            windows::show(ui, pane, |ui| self.ui_pane(ui, pane, is_dark));
        }
    }

    fn ui_pane(&mut self, ui: &mut egui::Ui, pane: Pane, is_dark: bool) {
        match pane {
            Pane::FreeEnergy => match &mut self.curves {
                Some(curves) => curves.ui(ui, &mut self.shown, is_dark),
                None => placeholder(ui, "Load structures in Onebody and run a scan"),
            },
            Pane::Thermodynamics => match &self.result {
                Some(result) => thermo::thermodynamics(ui, result),
                None => placeholder(ui, "Available once the scan finishes"),
            },
            Pane::Poses => {
                let result = self.result.as_ref();
                let shown = poses_shown(&self.status, result);
                self.poses.ui(ui, result.filter(|_| shown));
            }
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
        }

        section(ui);

        // --- Scan range ---
        ui.strong("Scan range");
        ui.add_space(2.0);

        egui::Grid::new("scan_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("R");
                ui.add(
                    egui::RangeSlider::new(&mut self.inputs.rmin, &mut self.inputs.rmax, R_SLIDER)
                        .min_separation(1.0)
                        .clamping(egui::SliderClamping::Never)
                        .suffix(" \u{00C5}"),
                )
                .on_hover_text(
                    "Sampled mass-centre separations. Below the lower end we assume \
                    infinite repulsion in the osmotic second virial coefficient, \
                    B\u{2082}, which directly affects the reduced virial coefficient, so \
                    pay attention when comparing with experiment or other models. \
                    Beyond the upper end the tail is fitted. Type a value to go past \
                    the slider's ends.",
                );
                ui.end_row();

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
                self.cancel_flag.store(true, Ordering::Relaxed);
                self.abandon_scan();
            }
            _ => {}
        }
        self.ui_status(ui, blocked);
    }

    /// Why a scan can't start, if it can't.
    fn start_blocker(&self, structures: &Structures) -> Option<&'static str> {
        if !self.gpu_available {
            Some(NO_GPU)
        } else if self.pair(structures).is_none() {
            Some("Pick both molecules from Onebody's structures")
        } else if self.inputs.rmin >= self.inputs.rmax {
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
        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
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
    fn pair<'a>(&self, structures: &'a Structures) -> Option<[&'a StructureFile; 2]> {
        let [a, b] = self.selected;
        Some([structures.get(a?)?, structures.get(b?)?])
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

    fn start_scan(
        &mut self,
        structures: &Structures,
        conditions: &SolutionConditions,
        forcefield: &ForceField,
    ) {
        let Some([mol1, mol2]) = self.pair(structures) else {
            return;
        };
        let (mol1_data, mol2_data) = (mol1.contents.clone(), mol2.contents.clone());
        self.scanned = self.selected[0].zip(self.selected[1]).map(<[_; 2]>::from);

        self.error = None;
        self.result = None;
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.status = ComputeStatus::Running {
            current_r: 0,
            total_r: 1,
        };

        // Empty until the first R-bin arrives; the plot then grows with the scan.
        self.curves = Some(Curves::new());

        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);
        let (prog_tx, prog_rx) = mpsc::channel();
        self.progress_rx = Some(prog_rx);
        // Checked after every R-bin, which is how Stop ends the scan.
        let cancel = self.cancel_flag.clone();
        let progress = move |p: ScanProgress| {
            let _ = prog_tx.send(p);
            if cancel.load(Ordering::Relaxed) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };

        let settings = self.inputs.settings(conditions, forcefield);

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let result =
                    crate::compute::run_scan_blocking(settings, mol1_data, mol2_data, progress);
                let _ = tx.send(result);
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let result =
                    crate::compute::run_scan_async(settings, mol1_data, mol2_data, progress).await;
                let _ = tx.send(result);
            });
        }
    }

    /// Poll for progress updates from the scan. Called each frame.
    fn poll_progress(&mut self) {
        let Some(rx) = &self.progress_rx else {
            return;
        };
        for p in rx.try_iter() {
            self.status = ComputeStatus::Running {
                current_r: p.done,
                total_r: p.total,
            };
            if let (Some(point), Some(curves)) = (p.point, &mut self.curves) {
                curves.push(point);
            }
        }
    }

    /// Poll for background scan completion. Called each frame.
    fn poll_result(&mut self) {
        if let Some(rx) = &self.result_rx {
            match rx.try_recv() {
                Ok(Ok(result)) => {
                    self.install_result(result);
                    self.status = ComputeStatus::Idle;
                    self.result_rx = None;
                    self.progress_rx = None;
                }
                Ok(Err(e)) => {
                    self.error = Some(format!("{e:#}"));
                    self.abandon_scan();
                }
                Err(mpsc::TryRecvError::Empty) => {} // still running
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.error = Some("Scan thread terminated unexpectedly".to_string());
                    self.abandon_scan();
                }
            }
        }
    }

    /// Ends a scan that won't deliver a result. Drops the partial curve, which would
    /// otherwise look like a finished PMF.
    fn abandon_scan(&mut self) {
        self.status = ComputeStatus::Idle;
        self.result_rx = None;
        self.progress_rx = None;
        self.curves = None;
    }

    /// Shared by finished scans and imports, so both show a result the same way.
    fn install_result(&mut self, result: ComputeResult) {
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
        matches!(self.status, ComputeStatus::Idle)
    }

    pub fn gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// File ▸ Export items: the shown scan's results as plain files.
    pub fn export_menu(&self, ui: &mut egui::Ui) {
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

    /// The pair the shown result was scanned from, if known; a running scan
    /// already names its pair, but shows no result yet.
    pub fn scanned(&self) -> Option<[StructureId; 2]> {
        self.result.as_ref().and(self.scanned)
    }

    fn download_csv(&self) {
        let Some(result) = &self.result else { return };
        let csv = duello::report::pmf_csv(&result.points);
        self.download_string(&csv, "pmf.csv");
    }
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
    let name = |id| structures.get(id).map(|f| f.name.as_str());
    egui::ComboBox::from_id_salt(label)
        .selected_text(slot.and_then(name).unwrap_or("—"))
        .show_ui(ui, |ui| {
            for (id, name) in structures.list() {
                ui.selectable_value(slot, Some(id), name);
            }
        })
        .response
        .on_hover_text(
            "A structure loaded in Onebody. Picking the same one for both molecules \
            scans a homo-dimer, which Manybody needs.",
        );
}
