use crate::manybody::ManybodyTab;
use crate::onebody::OnebodyTab;
use crate::twobody::TwobodyTab;

pub struct DuelloApp {
    tab: Tab,
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
        Self {
            tab: Tab::Twobody,
            onebody: OnebodyTab,
            twobody: TwobodyTab::new(),
            manybody: Default::default(),
        }
    }

    fn ui_tab_strip(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Onebody, "Onebody")
                .on_hover_text("Properties of a single molecule");
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
        self.twobody.poll(&ctx);
        egui::Panel::top("tabs").show(ui, |ui| self.ui_tab_strip(ui));
        self.manybody.tick();
        match self.tab {
            Tab::Onebody => self.onebody.ui(ui),
            Tab::Twobody => self.twobody.ui(ui),
            Tab::Manybody => self.manybody.ui(ui, self.twobody.archive()),
        }
        // A running many-body simulation advances once per frame.
        if self.manybody.is_running() {
            ctx.request_repaint();
        }
    }
}
