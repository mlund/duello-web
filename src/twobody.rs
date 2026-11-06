use crate::compute::{ComputeResult, ComputeStatus};
#[cfg(target_arch = "wasm32")]
use crate::files::download_bytes_as_file;
use crate::files::save_file;
use crate::inputs::{CgPolicy, Model, ScanInputs};
use duello::web_api::{PmfPoint, ScanProgress};
use duello::Subdivision;
use egui_plot::{Line, MarkerShape, Plot, PlotPoint, Points};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

/// Kept on screen while it applies, since it explains why "Run scan" is disabled.
const NO_GPU: &str =
    "No GPU/WebGPU adapter found. A WebGPU-capable browser is required (Chrome 113+, Edge 113+).";

fn subdivision_label(scheme: Subdivision) -> &'static str {
    match scheme {
        Subdivision::Geodesic => "Geodesic",
        Subdivision::Lattice => "Lattice",
    }
}

/// Action requested by mol_input_ui.
enum MolAction {
    None,
    PickFile,
    FetchPdb,
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

/// Per-molecule upload state.
struct MolUpload {
    name: String,
    data: Vec<u8>,
}

/// Two-body scan: inputs, the scan's lifecycle, PDB fetch, import/export
/// and the PMF plot. The app only needs its archive, for Manybody.
pub struct TwobodyTab {
    inputs: ScanInputs,
    mol1: Option<MolUpload>,
    mol2: Option<MolUpload>,
    /// PDB ID text fields for fetching from RCSB
    pdb_id1: String,
    pdb_id2: String,
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
    /// Channel for receiving file uploads (mol_num, name, data).
    file_rx: Option<mpsc::Receiver<(u8, String, Vec<u8>)>>,
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
            mol1: None,
            mol2: None,
            pdb_id1: "2CDS".to_string(),
            pdb_id2: "2CDS".to_string(),
            status: ComputeStatus::Idle,
            result: None,
            plot_cache: None,
            plot_needs_reset: false,
            error: (!gpu_available).then(|| NO_GPU.to_string()),
            gpu_available,
            result_rx: None,
            progress_rx: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            file_rx: None,
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
        self.poll_file_upload();
        self.poll_import();
        if matches!(self.status, ComputeStatus::Running { .. }) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let is_dark = ui.visuals().dark_mode;
        egui::Panel::left("inputs").min_size(220.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.ui_left_panel(ui);
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

    fn ui_left_panel(&mut self, ui: &mut egui::Ui) {
        // The browser shows this in the HTML header instead.
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::version::ui(ui);
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(2.0);
        }

        // --- Molecules ---
        ui.strong("Molecules");
        ui.add_space(2.0);

        ui.label("Molecule 1:");
        match Self::mol_input_ui(ui, &self.mol1, &mut self.pdb_id1) {
            MolAction::PickFile => self.pick_file(1),
            MolAction::FetchPdb => self.fetch_pdb(1),
            MolAction::None => {}
        }

        ui.add_space(2.0);
        ui.label("Molecule 2:");
        match Self::mol_input_ui(ui, &self.mol2, &mut self.pdb_id2) {
            MolAction::PickFile => self.pick_file(2),
            MolAction::FetchPdb => self.fetch_pdb(2),
            MolAction::None => {}
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(2.0);

        // --- Force field ---
        ui.strong("Force field");
        ui.add_space(2.0);

        egui::Grid::new("ff_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Model");
                egui::ComboBox::from_id_salt("model_combo")
                    .selected_text(self.inputs.model.label())
                    .show_ui(ui, |ui| {
                        for model in Model::ALL {
                            ui.selectable_value(&mut self.inputs.model, model, model.label());
                        }
                    })
                    .response
                    .on_hover_text("Residue-level force field for the short-range pair energy");
                ui.end_row();

                ui.label("CG policy");
                egui::ComboBox::from_id_salt("cg_combo")
                    .selected_text(match self.inputs.cg {
                        CgPolicy::Single => "Single bead",
                        CgPolicy::Multi => "Multi bead",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.inputs.cg, CgPolicy::Single, "Single bead");
                        ui.selectable_value(&mut self.inputs.cg, CgPolicy::Multi, "Multi bead");
                    })
                    .response
                    .on_hover_text("One bead per residue, or several per residue");
                ui.end_row();
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(2.0);

        // --- Solution ---
        ui.strong("Solution");
        ui.add_space(2.0);

        egui::Grid::new("solution_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("pH");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.ph)
                        .speed(0.1)
                        .range(0.0..=14.0),
                )
                .on_hover_text(
                    "Determines the average ionization state of titratable groups \
                    and N/C terminal ends using Metropolis GCMC swap moves.",
                );
                ui.end_row();

                ui.label("Ionic strength");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.molarity)
                        .speed(0.01)
                        .suffix(" M")
                        .range(0.0..=10.0),
                )
                .on_hover_text(
                    "Sets the Debye screening length used in both the Monte Carlo \
                    titration of ionizable groups and in the intermolecular \
                    protein-protein interaction energy via the Yukawa potential.",
                );
                ui.end_row();

                ui.label("Temperature");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.temperature)
                        .speed(1.0)
                        .suffix(" K")
                        .range(1.0..=1000.0),
                )
                .on_hover_text(
                    "Affects the thermal energy used in Boltzmann averaging \
                    and the dielectric constant of water.",
                );
                ui.end_row();
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(2.0);

        // --- Scan range ---
        ui.strong("Scan range");
        ui.add_space(2.0);

        egui::Grid::new("scan_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("R min");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.rmin)
                        .speed(0.5)
                        .suffix(" \u{00C5}"),
                )
                .on_hover_text(
                    "Minimum sampled center of mass separation between the proteins. \
                    Below this value we assume infinite repulsion in the osmotic second \
                    virial coefficient, B\u{2082}. Directly affects the reduced virial \
                    coefficient so pay attention when comparing with experiment or other models.",
                );
                ui.end_row();

                ui.label("R max");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.rmax)
                        .speed(0.5)
                        .suffix(" \u{00C5}"),
                )
                .on_hover_text(
                    "Largest sampled mass-centre separation; beyond it the tail is fitted",
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

        self.ui_angular_mesh(ui);

        ui.add_space(6.0);

        // --- Advanced (collapsed) ---
        egui::CollapsingHeader::new("Advanced")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("advanced_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
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
            validation_errors.push("R min must be less than R max");
        }
        if self.inputs.dr <= 0.0 {
            validation_errors.push("\u{0394}R must be positive");
        }

        for err in &validation_errors {
            ui.colored_label(egui::Color32::LIGHT_RED, *err);
        }

        let can_run = self.mol1.is_some()
            && (self.inputs.homo_dimer || self.mol2.is_some())
            && matches!(self.status, ComputeStatus::Idle)
            && validation_errors.is_empty()
            && self.gpu_available;

        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_run, egui::Button::new("Run scan"))
                .on_hover_text("Tabulate the pair energy over separations and orientations")
                .on_disabled_hover_text(
                    "Needs both molecules (or one for a homo-dimer), valid inputs, a GPU \
                    and no scan running",
                )
                .clicked()
            {
                self.start_scan();
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
            let idle = matches!(self.status, ComputeStatus::Idle);
            if ui
                .add_enabled(idle, egui::Button::new("Import scan..."))
                .on_hover_text("Load an exported .duello scan instead of running one")
                .clicked()
            {
                self.pick_scan_file();
            }
        });
    }

    /// Scheme and finest level of the orientation mesh.
    fn ui_angular_mesh(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
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
        });
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
                ui.label("Upload PDB files and run a scan to see the PMF plot.");
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

            // Download buttons
            let mut export = false;
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("PMF (.csv)")
                    .on_hover_text("Free energy and mean energy against separation")
                    .clicked()
                {
                    self.download_csv();
                }
                if ui
                    .button("Mol 1 (.xyz)")
                    .on_hover_text("Coarse-grained beads of molecule 1")
                    .clicked()
                {
                    self.download_string(&result.bodies[0].to_xyz(), "mol1_cg.xyz");
                }
                if ui
                    .button("Mol 2 (.xyz)")
                    .on_hover_text("Coarse-grained beads of molecule 2")
                    .clicked()
                {
                    self.download_string(&result.bodies[1].to_xyz(), "mol2_cg.xyz");
                }
                if ui
                    .button("Topology (.yaml)")
                    .on_hover_text("Bead kinds and pair potentials used by the scan")
                    .clicked()
                {
                    self.download_string(&result.topology_yaml, "topology.yaml");
                }
                export = ui
                    .button("Export scan (.duello)")
                    .on_hover_text("Table, CG bodies and results; re-import without rescanning")
                    .clicked();
            });
            // After the row: the closure holds `&self`, but exporting needs
            // `&mut self` to report failures.
            if export {
                self.export_scan();
            }
        }
    }

    /// UI for a single molecule input: upload button + PDB ID text field.
    /// Returns an action to perform after the UI is drawn (avoids borrow issues).
    fn mol_input_ui(ui: &mut egui::Ui, mol: &Option<MolUpload>, pdb_id: &mut String) -> MolAction {
        if let Some(m) = mol {
            ui.label(format!("Loaded: {}", m.name));
        }

        let mut action = MolAction::None;
        ui.horizontal(|ui| {
            if ui
                .button("Upload file...")
                .on_hover_text("Load a PDB file from disk")
                .clicked()
            {
                action = MolAction::PickFile;
            }
            ui.label("or");
            ui.add(
                egui::TextEdit::singleline(pdb_id)
                    .hint_text("PDB ID")
                    .desired_width(50.0),
            )
            .on_hover_text("Four-character RCSB PDB code, e.g. 4LZT");
            let id_valid = pdb_id.len() == 4 && pdb_id.chars().all(|c| c.is_alphanumeric());
            if ui
                .add_enabled(id_valid, egui::Button::new("Fetch"))
                .on_hover_text("Download this structure from RCSB")
                .on_disabled_hover_text("Enter a four-character PDB ID")
                .clicked()
            {
                action = MolAction::FetchPdb;
            }
        });
        action
    }

    /// Fetch a PDB file from RCSB by ID.
    fn fetch_pdb(&mut self, mol_num: u8) {
        let pdb_id = if mol_num == 1 {
            self.pdb_id1.trim().to_uppercase()
        } else {
            self.pdb_id2.trim().to_uppercase()
        };
        if pdb_id.len() != 4 {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.file_rx = Some(rx);

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let url = format!("https://files.rcsb.org/download/{pdb_id}.pdb");
                match fetch_url(&url).await {
                    Ok(data) => {
                        let _ = tx.send((mol_num, format!("{pdb_id}.pdb"), data));
                    }
                    Err(e) => {
                        log::error!("Failed to fetch PDB {pdb_id}: {e}");
                    }
                }
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let url = format!("https://files.rcsb.org/download/{pdb_id}.pdb");
                match ureq::get(&url).call() {
                    Ok(response) => {
                        let mut data = Vec::new();
                        if std::io::Read::read_to_end(
                            &mut response.into_body().as_reader(),
                            &mut data,
                        )
                        .is_ok()
                        {
                            let _ = tx.send((mol_num, format!("{pdb_id}.pdb"), data));
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to fetch PDB {pdb_id}: {e}");
                    }
                }
            });
        }
    }

    fn pick_file(&mut self, mol_num: u8) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PDB files", &["pdb"])
                .pick_file()
            {
                if let Ok(data) = std::fs::read(&path) {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let upload = MolUpload { name, data };
                    if mol_num == 1 {
                        self.mol1 = Some(upload);
                    } else {
                        self.mol2 = Some(upload);
                    }
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let (tx, rx) = mpsc::channel();
            self.file_rx = Some(rx);
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(file) = rfd::AsyncFileDialog::new()
                    .add_filter("PDB files", &["pdb"])
                    .pick_file()
                    .await
                {
                    let name = file.file_name();
                    let data = file.read().await;
                    let _ = tx.send((mol_num, name, data));
                }
            });
        }
    }

    fn start_scan(&mut self) {
        let mol1_data = match &self.mol1 {
            Some(m) => m.data.clone(),
            None => return,
        };
        let mol2_data = if self.inputs.homo_dimer {
            mol1_data.clone()
        } else {
            match &self.mol2 {
                Some(m) => m.data.clone(),
                None => return,
            }
        };

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

        let inputs = self.inputs.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let result =
                    crate::compute::run_scan_blocking(&inputs, mol1_data, mol2_data, progress);
                let _ = tx.send(result);
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let result =
                    crate::compute::run_scan_async(inputs, mol1_data, mol2_data, progress).await;
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

    /// Poll for async file uploads. Called each frame.
    fn poll_file_upload(&mut self) {
        if let Some(rx) = &self.file_rx {
            if let Ok((mol_num, name, data)) = rx.try_recv() {
                let upload = MolUpload { name, data };
                if mol_num == 1 {
                    self.mol1 = Some(upload);
                } else {
                    self.mol2 = Some(upload);
                }
                self.file_rx = None;
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

    /// Native asks for the path first: serializing the table can take a while,
    /// and is wasted if the dialog is cancelled.
    fn export_scan(&mut self) {
        let Some(result) = &self.result else { return };
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

    fn pick_scan_file(&mut self) {
        const FILTER: (&str, &[&str]) = ("Duello scan", &["duello"]);
        // Native: blocking dialog, as `pick_file` does, so no channel is needed.
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

    /// Replaces the shown result and inputs with the imported scan's. The
    /// molecules are cleared: they no longer match what the panel describes.
    fn import_scan(&mut self, bytes: &[u8]) {
        // A running scan streams into `plot_cache`; an import would mix into it.
        if !matches!(self.status, ComputeStatus::Idle) {
            self.error = Some("Import ignored: a scan is running".into());
            return;
        }
        let result = match ComputeResult::from_bytes(bytes) {
            Ok(result) => result,
            Err(e) => return self.error = Some(format!("Import failed: {e:#}")),
        };
        let model = &result.settings.model;
        let warning = Model::from_name(model)
            .is_none()
            .then(|| format!("Imported scan uses model '{model}', which the panel can't select"));
        self.error = warning.or_else(|| (!self.gpu_available).then(|| NO_GPU.to_string()));
        self.inputs = ScanInputs::from_settings(&result.settings, self.inputs.model);
        // Molecules, including uploads or fetches still in flight, no longer
        // match the panel, which now describes the imported scan.
        self.mol1 = None;
        self.mol2 = None;
        self.file_rx = None;
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

/// Fetch a URL and return its body as bytes (WASM only).
#[cfg(target_arch = "wasm32")]
async fn fetch_url(url: &str) -> anyhow::Result<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| anyhow::anyhow!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("not a Response"))?;
    if !resp.ok() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    let buf = JsFuture::from(
        resp.array_buffer()
            .map_err(|_| anyhow::anyhow!("no array_buffer"))?,
    )
    .await
    .map_err(|e| anyhow::anyhow!("array_buffer failed: {e:?}"))?;
    let array = js_sys::Uint8Array::new(&buf);
    Ok(array.to_vec())
}
