use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use crate::manybody::ManybodyTab;
use crate::onebody::OnebodyTab;
use crate::structures::Structures;
use crate::twobody::TwobodyTab;
use duello::archive::Session;
use duello::web_api::ScanSettings;

pub struct DuelloApp {
    tab: Tab,
    /// Inputs shared by the tabs; Onebody edits them.
    structures: Structures,
    conditions: SolutionConditions,
    forcefield: ForceField,
    onebody: OnebodyTab,
    twobody: TwobodyTab,
    manybody: ManybodyTab,
}

/// Single-protein analysis, two-body scan, or many-body MC built on the scan.
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Onebody,
    Twobody,
    Manybody,
}

/// egui's bundled fonts lack math glyphs such as ⟨⟩ and ⁻¹, which then show
/// as boxes. A DejaVu Sans subset (Latin, Greek, scripts, arrows, math;
/// license in assets/) fills the gaps without changing the default look.
fn install_fallback_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "dejavu".into(),
        egui::FontData::from_static(include_bytes!("../assets/DejaVuSans-subset.ttf")).into(),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("dejavu".into());
    }
    ctx.set_fonts(fonts);
}

impl DuelloApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fallback_font(&cc.egui_ctx);
        crate::viewer::register(cc);
        Self {
            tab: Tab::Onebody,
            structures: Structures::default(),
            conditions: SolutionConditions::default(),
            forcefield: ForceField::default(),
            onebody: OnebodyTab::default(),
            twobody: TwobodyTab::new(),
            manybody: Default::default(),
        }
    }

    /// Every tab's inputs, saved with the shown scan; the inverse of
    /// `restore`. Structures removed since the scan leave its pair unknown.
    fn session(&self) -> Session {
        let structures = &self.structures;
        let scanned = (self.twobody.scanned())
            .and_then(|[a, b]| Some([structures.index_of(a)?, structures.index_of(b)?]));
        let settings = self.twobody.archive().map(|a| &a.settings);
        Session {
            structures: structures.files(),
            scanned,
            estimate: settings.and_then(|s| self.conditions.estimate_for(s)),
            run: Some(self.manybody.spec(self.conditions.protein_molarity())),
        }
    }

    /// Every tab's inputs from an imported scan. A CLI scan carries no
    /// structures, which leaves Onebody's list as it is.
    fn restore(&mut self, settings: &ScanSettings, session: Session) {
        self.conditions.restore(settings, &session);
        if let Some(warning) = self.forcefield.adopt(settings) {
            self.twobody.warn(warning);
        }
        if let Some(run) = &session.run {
            self.manybody.restore(run, self.twobody.archive());
        }
        if session.structures.is_empty() {
            return;
        }
        let ids = self.structures.replace(session.structures);
        // Indices were checked against the structures on load.
        if let Some([a, b]) = session.scanned {
            self.twobody.select_scanned([ids[a], ids[b]]);
        }
    }

    /// File menu, tabs, and on the right what is running.
    fn ui_top_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| self.ui_file_menu(ui));
            ui.separator();
            self.ui_tab_strip(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.ui_status(ui)
            });
        });
    }

    fn ui_file_menu(&mut self, ui: &mut egui::Ui) {
        let has_scan = self.twobody.archive().is_some();
        if ui
            .add_enabled(self.twobody.idle(), egui::Button::new("Open scan…"))
            .on_hover_text("Load a .duello scan and restore every tab's inputs saved with it")
            .on_disabled_hover_text("Not while a scan is running")
            .clicked()
        {
            self.twobody.open_scan();
        }
        if ui
            .add_enabled(has_scan, egui::Button::new("Save scan…"))
            .on_hover_text(
                "Table, CG bodies and results, plus every tab's inputs (structures \
                 and their names, force field, conditions, Manybody), as .duello",
            )
            .on_disabled_hover_text("Run or open a scan first")
            .clicked()
        {
            let session = self.session();
            self.twobody.export_scan(session);
        }
        ui.separator();
        ui.add_enabled_ui(has_scan, |ui| {
            ui.menu_button("Export", |ui| self.twobody.export_menu(ui))
                .response
                .on_hover_text("The shown scan's results as plain files")
                .on_disabled_hover_text("Run or open a scan first");
        });
    }

    fn ui_tab_strip(&mut self, ui: &mut egui::Ui) {
        ui.selectable_value(&mut self.tab, Tab::Onebody, "Onebody")
            .on_hover_text("Structures, force field and solution conditions shared by every tab");
        ui.selectable_value(&mut self.tab, Tab::Twobody, "Twobody")
            .on_hover_text("Orientation-averaged potential of mean force between two molecules");
        ui.selectable_value(&mut self.tab, Tab::Manybody, "Manybody")
            .on_hover_text("Monte Carlo of many copies of the Twobody scan's molecule");
    }

    /// Right to left: GPU, then (natively) the version, which the browser
    /// shows in its HTML header instead.
    fn ui_status(&self, ui: &mut egui::Ui) {
        if self.twobody.gpu_available() {
            ui.weak("GPU ✓")
                .on_hover_text("A GPU/WebGPU adapter is available for scans");
        } else {
            ui.colored_label(egui::Color32::LIGHT_RED, "No GPU")
                .on_hover_text("Scans need WebGPU (Chrome 113+, Edge 113+)");
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::version::ui(ui);
    }
}

impl eframe::App for DuelloApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.structures.poll();
        self.twobody.poll(&ctx);
        egui::Panel::top("tabs").show(ui, |ui| self.ui_top_bar(ui));
        self.manybody.tick(self.twobody.archive());
        match self.tab {
            Tab::Onebody => {
                // Z and mass for the estimate, from a molecule Manybody can use.
                let homo = self.twobody.archive().filter(|a| a.settings.homo_dimer);
                let body = homo.map(|a| &a.bodies[0]);
                self.onebody.ui(
                    ui,
                    &mut self.structures,
                    &mut self.conditions,
                    &mut self.forcefield,
                    body,
                )
            }
            Tab::Twobody => {
                self.twobody
                    .ui(ui, &self.structures, &self.conditions, &self.forcefield)
            }
            Tab::Manybody => self
                .manybody
                .ui(ui, self.twobody.archive(), &self.conditions),
        }
        // After the tabs: a browser import lands while Twobody polls.
        if let Some((settings, session)) = self.twobody.take_import() {
            // The imported scan shows in Twobody; a cancelled dialog stays put.
            self.tab = Tab::Twobody;
            self.restore(&settings, session);
            ctx.request_repaint();
        }
        // A busy many-body tab (running, GPU setup, late frames) needs frames.
        if self.manybody.is_busy() {
            ctx.request_repaint();
        }
    }
}
