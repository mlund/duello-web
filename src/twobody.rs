use crate::compute::{ComputeResult, ComputeStatus};
use crate::conditions::SolutionConditions;
#[cfg(target_arch = "wasm32")]
use crate::files::download_bytes_as_file;
use crate::files::save_file;
use crate::forcefield::ForceField;
use crate::inputs::ScanInputs;
use crate::structures::{StructureId, Structures};
use crate::widgets::section;
use duello::archive::{Session, StructureFile};
use duello::web_api::{PmfPoint, ScanProgress, ScanSettings};
use duello::Subdivision;
use egui_plot::{Line, MarkerShape, Plot, PlotPoint, Points};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

/// Kept on screen while it applies, since it explains why "Run scan" is disabled.
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

/// Plot data in egui_plot's own type, so frames borrow it instead of converting.
/// Filled point by point while a scan runs, then replaced by the final result.
#[derive(Default)]
struct PlotCache {
    pmf: Vec<PlotPoint>,
    mean_energy: Vec<PlotPoint>,
    /// -TS = F - U, the entropic part of the free energy.
    entropy: Vec<PlotPoint>,
}

impl PlotCache {
    fn from_result(result: &ComputeResult) -> Self {
        let mut cache = Self::default();
        result.points.iter().for_each(|&p| cache.push(p));
        cache
    }

    fn push(&mut self, p: PmfPoint) {
        let r = p.r as f64;
        let (f, u) = (p.free_energy as f64, p.mean_energy as f64);
        self.pmf.push(PlotPoint::new(r, f));
        self.mean_energy.push(PlotPoint::new(r, u));
        self.entropy.push(PlotPoint::new(r, f - u));
    }
}

/// Two-body scan of a pair picked from Onebody's structures: inputs, the
/// scan's lifecycle, import/export and the PMF plot.
pub struct TwobodyTab {
    inputs: ScanInputs,
    /// Molecules 1 and 2; the same entry twice is a homo-dimer.
    selected: [Option<StructureId>; 2],
    /// The pair the shown result was scanned from, for export; None when
    /// unknown, e.g. imported from the CLI.
    scanned: Option<[StructureId; 2]>,
    /// A scan just imported, until the app restores its session.
    imported: Option<(ScanSettings, Session)>,
    status: ComputeStatus,
    result: Option<ComputeResult>,
    /// Cached plot data (computed once from result, reused each frame).
    plot_cache: Option<PlotCache>,
    /// Set to true when new results arrive, consumed on next plot render.
    plot_needs_reset: bool,
    error: Option<String>,
    gpu_available: bool,
    /// Channel for receiving results from background scan thread.
    result_rx: Option<mpsc::Receiver<anyhow::Result<ComputeResult>>>,
    /// Progress and PMF points, one message per finished R-bin.
    progress_rx: Option<mpsc::Receiver<ScanProgress>>,
    /// Flag to signal cancellation to the running scan.
    cancel_flag: Arc<AtomicBool>,
    /// Bytes of a scan archive picked for import.
    import_rx: Option<mpsc::Receiver<Vec<u8>>>,
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
            imported: None,
            status: ComputeStatus::Idle,
            result: None,
            plot_cache: None,
            plot_needs_reset: false,
            error: (!gpu_available).then(|| NO_GPU.to_string()),
            gpu_available,
            result_rx: None,
            progress_rx: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            import_rx: None,
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
        self.poll_import();
        if matches!(self.status, ComputeStatus::Running { .. }) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    /// A scan just imported, once: its settings and saved session, for
    /// the app to restore into every tab.
    pub fn take_import(&mut self) -> Option<(ScanSettings, Session)> {
        self.imported.take()
    }

    /// Shown in the panel, e.g. why an import couldn't restore everything.
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

        egui::Panel::bottom("results")
            .min_size(80.0)
            .show(ui, |ui| {
                self.ui_bottom_panel(ui);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.ui_center_panel(ui, is_dark);
        });
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

        // --- Validation & Run ---
        let mut validation_errors = Vec::new();
        if self.inputs.rmin >= self.inputs.rmax {
            validation_errors.push("The R range must have a lower end below its upper end");
        }
        if self.inputs.dr <= 0.0 {
            validation_errors.push("\u{0394}R must be positive");
        }

        for err in &validation_errors {
            ui.colored_label(egui::Color32::LIGHT_RED, *err);
        }

        let can_run = self.pair(structures).is_some()
            && self.idle()
            && validation_errors.is_empty()
            && self.gpu_available;

        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_run, egui::Button::new("Run scan"))
                .on_hover_text("Tabulate the pair energy over separations and orientations")
                .on_disabled_hover_text(
                    "Needs both molecules picked from Onebody's structures, valid inputs, \
                    a GPU and no scan running",
                )
                .clicked()
            {
                self.start_scan(structures, conditions, forcefield);
            }
            if matches!(self.status, ComputeStatus::Running { .. })
                && ui
                    .button("Cancel")
                    .on_hover_text("Abandon the running scan")
                    .clicked()
            {
                self.cancel_flag.store(true, Ordering::Relaxed);
                self.abandon_scan();
            }
        });
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

    fn ui_center_panel(&mut self, ui: &mut egui::Ui, is_dark: bool) {
        let Some(cache) = &self.plot_cache else {
            ui.centered_and_justified(|ui| {
                ui.label("Load structures in Onebody and run a scan to see the PMF plot.");
            });
            return;
        };
        let rgb = |dark: (u8, u8, u8), light: (u8, u8, u8)| {
            let (r, g, b) = if is_dark { dark } else { light };
            egui::Color32::from_rgb(r, g, b)
        };
        let series = [
            (
                "Free energy",
                &cache.pmf,
                rgb((100, 150, 255), (30, 60, 180)),
                MarkerShape::Circle,
            ),
            (
                "Average energy",
                &cache.mean_energy,
                rgb((255, 120, 100), (200, 40, 30)),
                MarkerShape::Diamond,
            ),
            (
                "Entropy (\u{2212}TS)",
                &cache.entropy,
                rgb((110, 200, 120), (30, 130, 50)),
                MarkerShape::Square,
            ),
        ];

        let mut plot = Plot::new("pmf_plot")
            .legend(egui_plot::Legend::default())
            .auto_bounds(egui::Vec2b::TRUE)
            .x_axis_label("Center of mass separation, R (\u{00C5})")
            .y_axis_label("Energy (kT)");
        // Reset only when a scan starts or ends; auto_bounds follows the growing curve
        // in between, so the user can still zoom while it runs.
        if self.plot_needs_reset {
            plot = plot.reset();
            self.plot_needs_reset = false;
        }
        plot.show(ui, |plot_ui| {
            for (name, data, color, shape) in series {
                let data = data.as_slice();
                plot_ui.line(Line::new(name, data).color(color).width(3.0_f32));
                // Unnamed, so the legend lists each series once.
                plot_ui.points(
                    Points::new("", data)
                        .color(color)
                        .shape(shape)
                        .radius(4.0_f32),
                );
            }
        });
    }

    fn ui_bottom_panel(&mut self, ui: &mut egui::Ui) {
        // Progress bar
        match &self.status {
            ComputeStatus::Running { current_r, total_r } => {
                let progress = *current_r as f32 / *total_r as f32;
                ui.add(
                    egui::ProgressBar::new(progress)
                        .text(format!("Step {}/{}", current_r, total_r)),
                );
            }
            ComputeStatus::Idle => {}
        }

        // Error display
        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
        }

        // Results
        if let Some(result) = &self.result {
            ui.separator();
            let (mw1, mw2) = result.molar_masses();
            let (z1, z2) = result.net_charges();
            ui.label(format!("Mw = {mw1:.1} / {mw2:.1} g/mol"));
            ui.label(format!("Net charge = {z1:+.1} / {z2:+.1} e"));
            ui.separator();
            let b2 = result.virial.b2();
            ui.label(format!("B\u{2082} = {b2:.2} \u{00C5}\u{00B3}"));
            ui.label(format!(
                "B\u{2082} = {:.2e} mol*ml/g\u{00B2}",
                result.virial.mol_ml_per_gram2(mw1, mw2)
            ));
            ui.label(format!(
                "B\u{2082}/B\u{2082}hs = {:.2}, \u{03C3} = {:.2} \u{00C5}",
                result.virial.reduced(),
                result.virial.sigma()
            ));
            if let Some(kd) = result.virial.dissociation_const() {
                ui.label(format!(
                    "K\u{1D451} = {kd:.2e} mol/l, \u{03C3} = {:.2} \u{00C5}",
                    result.virial.sigma()
                ));
            }
        }
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
        self.cancel_flag.store(false, Ordering::Relaxed);
        self.status = ComputeStatus::Running {
            current_r: 0,
            total_r: 1,
        };

        // Empty until the first R-bin arrives; the plot then grows with the scan.
        self.plot_cache = Some(PlotCache::default());
        self.plot_needs_reset = true;

        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);
        let (prog_tx, prog_rx) = mpsc::channel();
        self.progress_rx = Some(prog_rx);
        // Checked after every R-bin, which is how Cancel stops the scan.
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
            if let (Some(point), Some(cache)) = (p.point, &mut self.plot_cache) {
                cache.push(point);
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
        self.plot_cache = None;
    }

    /// Shared by finished scans and imports, so both show a result the same way.
    fn install_result(&mut self, result: ComputeResult) {
        self.plot_cache = Some(PlotCache::from_result(&result));
        self.plot_needs_reset = true;
        self.result = Some(result);
    }

    /// Save failures only reach the log, as before the scan export existed.
    fn download_string(&self, content: &str, filename: &str) {
        if let Err(e) = save_file(content.as_bytes(), filename) {
            log::error!("Failed to save {filename}: {e}");
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

    /// The pair the shown result was scanned from, if known.
    pub fn scanned(&self) -> Option<[StructureId; 2]> {
        self.scanned
    }

    /// Writes the shown scan with `session`, every tab's inputs. Native
    /// asks for the path first: serializing the table can take a while, and
    /// is wasted if the dialog is cancelled.
    pub fn export_scan(&mut self, session: Session) {
        let Some(result) = &mut self.result else {
            return;
        };
        result.session = session;
        const NAME: &str = "scan.duello";
        #[cfg(not(target_arch = "wasm32"))]
        let Some(path) = rfd::FileDialog::new().set_file_name(NAME).save_file() else {
            return;
        };
        let bytes = match result.to_bytes() {
            Ok(bytes) => bytes,
            Err(e) => return self.error = Some(format!("Export failed: {e:#}")),
        };
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = std::fs::write(&path, bytes) {
            self.error = Some(format!("Export failed: {e}"));
        }
        #[cfg(target_arch = "wasm32")]
        download_bytes_as_file(&bytes, NAME);
    }

    /// File ▸ Open: picks a `.duello` and imports it.
    pub fn open_scan(&mut self) {
        const FILTER: (&str, &[&str]) = ("Duello scan", &["duello"]);
        // Native: a blocking dialog, so no channel is needed.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(FILTER.0, FILTER.1)
            .pick_file()
        {
            match std::fs::read(&path) {
                Ok(bytes) => self.import_scan(&bytes),
                Err(e) => self.error = Some(format!("Import failed: {e}")),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let (tx, rx) = mpsc::channel();
            self.import_rx = Some(rx);
            wasm_bindgen_futures::spawn_local(async move {
                let dialog = rfd::AsyncFileDialog::new().add_filter(FILTER.0, FILTER.1);
                if let Some(file) = dialog.pick_file().await {
                    let _ = tx.send(file.read().await);
                }
            });
        }
    }

    /// WASM reads the picked file asynchronously; a cancelled dialog drops the sender.
    fn poll_import(&mut self) {
        let Some(received) = self.import_rx.as_ref().map(|rx| rx.try_recv()) else {
            return;
        };
        match received {
            Ok(bytes) => {
                self.import_rx = None;
                self.import_scan(&bytes);
            }
            Err(mpsc::TryRecvError::Disconnected) => self.import_rx = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    /// Replaces the shown result and inputs with the imported scan's; its
    /// solution and session reach the other tabs via `take_import`.
    fn import_scan(&mut self, bytes: &[u8]) {
        // A running scan streams into `plot_cache`; an import would mix into it.
        if !self.idle() {
            self.error = Some("Import ignored: a scan is running".into());
            return;
        }
        let mut result = match ComputeResult::from_bytes(bytes) {
            Ok(result) => result,
            Err(e) => return self.error = Some(format!("Import failed: {e:#}")),
        };
        self.error = (!self.gpu_available).then(|| NO_GPU.to_string());
        self.inputs = ScanInputs::from_settings(&result.settings);
        let session = std::mem::take(&mut result.session);
        self.imported = Some((result.settings.clone(), session));
        self.scanned = None;
        self.install_result(result);
    }

    fn download_csv(&self) {
        let Some(result) = &self.result else { return };
        let mut csv = String::from("R/A,F/kT,U/kT\n");
        for p in &result.points {
            csv.push_str(&format!(
                "{:.2},{:.4},{:.4}\n",
                p.r, p.free_energy, p.mean_energy
            ));
        }
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
