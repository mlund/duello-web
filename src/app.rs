use crate::compute::{ComputeResult, ComputeStatus};
use egui_plot::{Line, MarkerShape, Plot, Points};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

/// Force field model choices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Calvados3,
    KimHummer,
    Pasquier,
}

impl Model {
    pub fn name(&self) -> &str {
        match self {
            Model::Calvados3 => "calvados3",
            Model::KimHummer => "kimhummer",
            Model::Pasquier => "pasquier",
        }
    }

    fn label(&self) -> &str {
        match self {
            Model::Calvados3 => "CALVADOS 3",
            Model::KimHummer => "Kim-Hummer",
            Model::Pasquier => "Pasquier",
        }
    }
}

/// Coarse-graining policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CgPolicy {
    Single,
    Multi,
}

/// Scan parameters editable in the UI.
#[derive(Clone)]
pub struct ScanInputs {
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub molarity: f64,
    pub temperature: f64,
    pub max_ndiv: usize,
    pub gradient_threshold: f64,
    pub cutoff: f64,
    pub grid_size: usize,
    pub ph: f64,
    pub model: Model,
    pub cg: CgPolicy,
    pub homo_dimer: bool,
}

impl Default for ScanInputs {
    fn default() -> Self {
        Self {
            rmin: 26.0,
            rmax: 60.0,
            dr: 1.0,
            molarity: 0.1,
            temperature: 298.15,
            max_ndiv: 2,
            gradient_threshold: 0.5,
            cutoff: 20.0,
            grid_size: 200,
            ph: 7.0,
            model: Model::Calvados3,
            cg: CgPolicy::Single,
            homo_dimer: false,
        }
    }
}

/// Action requested by mol_input_ui.
enum MolAction {
    None,
    PickFile,
    FetchPdb,
}

/// Pre-computed plot data to avoid per-frame conversion.
struct PlotCache {
    pmf: Vec<[f64; 2]>,
    mean_energy: Vec<[f64; 2]>,
}

impl PlotCache {
    fn from_result(result: &ComputeResult) -> Self {
        Self {
            pmf: result
                .pmf_data
                .iter()
                .map(|&(r, f)| [r as f64, f as f64])
                .collect(),
            mean_energy: result
                .mean_energy_data
                .iter()
                .map(|&(r, u)| [r as f64, u as f64])
                .collect(),
        }
    }
}

/// Per-molecule upload state.
struct MolUpload {
    name: String,
    data: Vec<u8>,
}

pub struct DuelloApp {
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
    /// Channel for receiving progress updates (current_r, total_r).
    progress_rx: Option<mpsc::Receiver<(usize, usize)>>,
    /// Flag to signal cancellation to the running scan.
    cancel_flag: Arc<AtomicBool>,
    /// Channel for receiving file uploads (mol_num, name, data).
    file_rx: Option<mpsc::Receiver<(u8, String, Vec<u8>)>>,
}

impl DuelloApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
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
            error: if gpu_available {
                None
            } else {
                Some("No GPU/WebGPU adapter found. A WebGPU-capable browser is required (Chrome 113+, Edge 113+).".to_string())
            },
            gpu_available,
            result_rx: None,
            progress_rx: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            file_rx: None,
        }
    }

    fn ui_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Duello");
        ui.hyperlink_to("duello on GitHub", "https://github.com/mlund/duello");
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(2.0);

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
                        ui.selectable_value(&mut self.inputs.model, Model::Calvados3, "CALVADOS 3");
                        ui.selectable_value(&mut self.inputs.model, Model::KimHummer, "Kim-Hummer");
                        ui.selectable_value(&mut self.inputs.model, Model::Pasquier, "Pasquier");
                    });
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
                    });
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
                );
                ui.end_row();

                ui.label("\u{0394}R");
                ui.add(
                    egui::DragValue::new(&mut self.inputs.dr)
                        .speed(0.1)
                        .suffix(" \u{00C5}")
                        .range(0.1..=10.0),
                );
                ui.end_row();
            });

        ui.add_space(4.0);

        // --- Resolution ---
        ui.label("Angular resolution");
        ui.add(egui::Slider::new(&mut self.inputs.max_ndiv, 0..=4))
            .on_hover_text(
                "Number of icosphere subdivisions controlling the angular resolution. \
                Zero means 12 vertices on the regular icosahedron; each subdivision \
                increases the number of angular samples.",
            );

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
                        );
                        ui.end_row();

                        ui.label("Gradient threshold");
                        ui.add(
                            egui::DragValue::new(&mut self.inputs.gradient_threshold)
                                .speed(0.1)
                                .range(0.0..=10.0),
                        );
                        ui.end_row();

                        ui.label("Spline grid size");
                        ui.add(
                            egui::DragValue::new(&mut self.inputs.grid_size)
                                .speed(10)
                                .range(50..=1000),
                        );
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
                .clicked()
            {
                self.start_scan();
            }
            if matches!(self.status, ComputeStatus::Running { .. }) && ui.button("Cancel").clicked()
            {
                self.cancel_flag.store(true, Ordering::Relaxed);
                self.status = ComputeStatus::Idle;
                self.result_rx = None;
                self.progress_rx = None;
            }
        });
    }

    fn ui_center_panel(&mut self, ui: &mut egui::Ui, is_dark: bool) {
        if let Some(cache) = &self.plot_cache {
            let blue = if is_dark {
                egui::Color32::from_rgb(100, 150, 255)
            } else {
                egui::Color32::from_rgb(30, 60, 180)
            };
            let red = if is_dark {
                egui::Color32::from_rgb(255, 120, 100)
            } else {
                egui::Color32::from_rgb(200, 40, 30)
            };

            let mut plot = Plot::new("pmf_plot")
                .legend(egui_plot::Legend::default())
                .auto_bounds(egui::Vec2b::TRUE)
                .x_axis_label("Center of mass separation, R (\u{00C5})")
                .y_axis_label("Energy (kT)");
            if self.plot_needs_reset {
                plot = plot.reset();
                self.plot_needs_reset = false;
            }
            plot
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Free energy", cache.pmf.clone())
                            .color(blue)
                            .width(2.0),
                    );
                    plot_ui.points(
                        Points::new("", cache.pmf.clone())
                            .color(blue)
                            .shape(MarkerShape::Circle)
                            .radius(3.0),
                    );

                    plot_ui.line(
                        Line::new("Average energy", cache.mean_energy.clone())
                            .color(red)
                            .width(2.0),
                    );
                    plot_ui.points(
                        Points::new("", cache.mean_energy.clone())
                            .color(red)
                            .shape(MarkerShape::Diamond)
                            .radius(3.0),
                    );
                });
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Upload PDB files and run a scan to see the PMF plot.");
            });
        }
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
            let (mw1, mw2) = result.molar_masses;
            let (z1, z2) = result.net_charges;
            ui.label(format!("Mw = {mw1:.1} / {mw2:.1} g/mol"));
            ui.label(format!("Net charge = {z1:+.1} / {z2:+.1} e"));
            ui.separator();
            let b2 = result.virial.b2();
            ui.label(format!("B\u{2082} = {b2:.2} \u{00C5}\u{00B3}"));
            let (mw1, mw2) = result.molar_masses;
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
            ui.horizontal_wrapped(|ui| {
                if ui.button("PMF (.csv)").clicked() {
                    self.download_csv();
                }
                if ui.button("Mol 1 (.xyz)").clicked() {
                    self.download_string(&result.cg_xyz.0, "mol1_cg.xyz");
                }
                if ui.button("Mol 2 (.xyz)").clicked() {
                    self.download_string(&result.cg_xyz.1, "mol2_cg.xyz");
                }
                if ui.button("Topology (.yaml)").clicked() {
                    self.download_string(&result.topology_yaml, "topology.yaml");
                }
            });
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
            if ui.button("Upload file...").clicked() {
                action = MolAction::PickFile;
            }
            ui.label("or");
            ui.add(
                egui::TextEdit::singleline(pdb_id)
                    .hint_text("PDB ID")
                    .desired_width(50.0),
            );
            let id_valid = pdb_id.len() == 4 && pdb_id.chars().all(|c| c.is_alphanumeric());
            if ui
                .add_enabled(id_valid, egui::Button::new("Fetch"))
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

        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);

        let inputs = self.inputs.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let result = crate::compute::run_scan_blocking(&inputs, mol1_data, mol2_data);
                let _ = tx.send(result);
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let (prog_tx, prog_rx) = mpsc::channel();
            self.progress_rx = Some(prog_rx);
            let cancel = self.cancel_flag.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = crate::compute::run_scan_async(
                    inputs,
                    mol1_data,
                    mol2_data,
                    move |current, total| {
                        let _ = prog_tx.send((current, total));
                    },
                    cancel,
                )
                .await;
                let _ = tx.send(result);
            });
        }
    }

    /// Poll for progress updates from the scan. Called each frame.
    fn poll_progress(&mut self) {
        if let Some(rx) = &self.progress_rx {
            // Drain all pending progress updates, keeping the latest
            while let Ok((current, total)) = rx.try_recv() {
                self.status = ComputeStatus::Running {
                    current_r: current,
                    total_r: total,
                };
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
                    self.plot_cache = Some(PlotCache::from_result(&result));
                    self.plot_needs_reset = true;
                    self.result = Some(result);
                    self.status = ComputeStatus::Idle;
                    self.result_rx = None;
                    self.progress_rx = None;
                }
                Ok(Err(e)) => {
                    self.error = Some(format!("{e:#}"));
                    self.status = ComputeStatus::Idle;
                    self.result_rx = None;
                    self.progress_rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => {} // still running
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.error = Some("Scan thread terminated unexpectedly".to_string());
                    self.status = ComputeStatus::Idle;
                    self.result_rx = None;
                    self.progress_rx = None;
                }
            }
        }
    }

    fn download_string(&self, content: &str, filename: &str) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(path) = rfd::FileDialog::new().set_file_name(filename).save_file() {
                if let Err(e) = std::fs::write(&path, content) {
                    log::error!("Failed to save {filename}: {e}");
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            download_string_as_file(content, filename);
        }
    }

    fn download_csv(&self) {
        let Some(result) = &self.result else { return };
        let mut csv = String::from("R/A,F/kT,U/kT\n");
        for (pmf, u) in result.pmf_data.iter().zip(&result.mean_energy_data) {
            csv.push_str(&format!("{:.2},{:.4},{:.4}\n", pmf.0, pmf.1, u.1));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(path) = rfd::FileDialog::new().set_file_name("pmf.csv").save_file() {
                if let Err(e) = std::fs::write(&path, &csv) {
                    log::error!("Failed to save CSV: {e}");
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            download_string_as_file(&csv, "pmf.csv");
        }
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

/// Trigger a browser file download from a string (WASM only).
#[cfg(target_arch = "wasm32")]
fn download_string_as_file(content: &str, filename: &str) {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let blob = web_sys::Blob::new_with_str_sequence(&js_sys::Array::of1(
        &wasm_bindgen::JsValue::from_str(content),
    ))
    .unwrap();
    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();
    let a: web_sys::HtmlAnchorElement = document.create_element("a").unwrap().dyn_into().unwrap();
    a.set_href(&url);
    a.set_download(filename);
    a.click();
    web_sys::Url::revoke_object_url(&url).unwrap();
}

impl eframe::App for DuelloApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_result();
        self.poll_progress();
        self.poll_file_upload();

        let ctx = ui.ctx().clone();
        let is_dark = ui.visuals().dark_mode;

        // Repaint periodically while a scan is running (not every frame)
        if matches!(self.status, ComputeStatus::Running { .. }) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        egui::Panel::left("inputs")
            .min_size(220.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_left_panel(ui);
                });
            });

        egui::Panel::bottom("results")
            .min_size(80.0)
            .show_inside(ui, |ui| {
                self.ui_bottom_panel(ui);
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.ui_center_panel(ui, is_dark);
        });
    }
}
