use crate::conditions::SolutionConditions;
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
            onebody: OnebodyTab,
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

    fn ui_tab_strip(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Onebody, "Onebody")
                .on_hover_text("Structures and solution conditions shared by every tab");
            ui.selectable_value(&mut self.tab, Tab::Twobody, "Twobody")
                .on_hover_text(
                    "Orientation-averaged potential of mean force between two molecules",
                );
            ui.selectable_value(&mut self.tab, Tab::Manybody, "Manybody")
                .on_hover_text("Monte Carlo of many copies of the Twobody scan's molecule");
        });
    }
}

impl eframe::App for DuelloApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.structures.poll();
        self.twobody.poll(&ctx);
        egui::Panel::top("tabs").show(ui, |ui| self.ui_tab_strip(ui));
        self.manybody.tick(self.twobody.archive());
        match self.tab {
            Tab::Onebody => {
                // Z and mass for the estimate, from a molecule Manybody can use.
                let homo = self.twobody.archive().filter(|a| a.settings.homo_dimer);
                let body = homo.map(|a| &a.bodies[0]);
                self.onebody
                    .ui(ui, &mut self.structures, &mut self.conditions, body)
            }
            Tab::Twobody => self.twobody.ui(ui, &self.structures, &self.conditions),
            Tab::Manybody => self
                .manybody
                .ui(ui, self.twobody.archive(), &self.conditions),
        }
        // After the tabs: a native import finishes inside Twobody's UI.
        if self.twobody.take_export_request() {
            let session = self.session();
            self.twobody.export_scan(session);
        }
        if let Some((settings, session)) = self.twobody.take_import() {
            self.restore(&settings, session);
            ctx.request_repaint();
        }
        // A busy many-body tab (running, GPU setup, late frames) needs frames.
        if self.manybody.is_busy() {
            ctx.request_repaint();
        }
    }
}
