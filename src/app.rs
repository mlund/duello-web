use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use crate::manybody::ManybodyTab;
use crate::manybody::Opened;
use crate::onebody::OnebodyTab;
use crate::session_file::SessionFile;
use crate::structures::Structures;
use crate::twobody::TwobodyTab;
use duello::archive::Session;
use duello::bundle::{Bundle, BundleRef, Extracts};
use duello::nbody::RunRecord;

pub struct DuelloApp {
    tab: Tab,
    /// Inputs shared by the tabs; Onebody edits them.
    structures: Structures,
    conditions: SolutionConditions,
    forcefield: ForceField,
    onebody: OnebodyTab,
    twobody: TwobodyTab,
    manybody: ManybodyTab,
    file: SessionFile,
    /// Why the last open or save failed, until dismissed.
    notice: Option<String>,
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
        Self::blank()
    }

    /// No structures, default inputs, no scan or run.
    fn blank() -> Self {
        Self {
            tab: Tab::Onebody,
            structures: Structures::default(),
            conditions: SolutionConditions::default(),
            forcefield: ForceField::default(),
            onebody: OnebodyTab::default(),
            twobody: TwobodyTab::new(),
            manybody: Default::default(),
            file: SessionFile::default(),
            notice: None,
        }
    }

    /// Every tab's inputs; the inverse of `restore`. Structures removed, or
    /// with other chains, since the scan leave its pair unknown.
    fn session(&self) -> Session {
        let structures = &self.structures;
        let scanned = (self.twobody.scanned(structures))
            .and_then(|[a, b]| Some([structures.index_of(a)?, structures.index_of(b)?]));
        let inputs = (self.twobody.inputs()).settings(&self.conditions, &self.forcefield);
        Session {
            inputs: Some(inputs),
            structures: structures.files(),
            scanned,
            estimate: self.conditions.estimate,
            run: Some(self.manybody.spec(self.conditions.protein_molarity())),
        }
    }

    /// Onebody's analysed molecule, which only the saved file shows.
    fn extracts(&self) -> Extracts<'_> {
        let selected = self.structures.selected();
        let file = selected.and_then(|id| Some((id, self.structures.get(id)?)));
        let molecule = file.and_then(|(id, file)| self.onebody.extract(id, file));
        Extracts {
            molecules: molecule.into_iter().collect(),
        }
    }

    /// What Save session writes, given `session` from [`Self::session`]
    /// and `run` from Manybody's results.
    fn bundle<'a>(
        &'a self,
        session: &'a Session,
        run: Option<&'a RunRecord>,
        trajectory: Option<&'a [u8]>,
    ) -> BundleRef<'a> {
        BundleRef {
            session,
            scan: self.twobody.archive(),
            run,
            trajectory,
        }
    }

    fn save_session(&mut self) {
        let zip = || {
            let results = self.manybody.results();
            let run = results.map(|r| r.record());
            let trajectory = results.and_then(|r| r.trajectory());
            let session = self.session();
            let bundle = self.bundle(&session, run.as_deref(), trajectory.as_deref());
            bundle.to_zip(&self.extracts())
        };
        if let Err(e) = SessionFile::save(zip) {
            self.notice = Some(format!("Save failed: {e:#}"));
        }
    }

    /// Takes a picked session once it is read; true if one was. A scan or
    /// run started while the browser read the file would mix with it.
    fn poll_open(&mut self) -> bool {
        let Some(opened) = self.file.take_opened() else {
            return false;
        };
        self.notice = match opened {
            Ok(_) if self.busy() => Some("Open ignored: a scan or run is active".into()),
            Ok(bundle) => {
                self.restore(bundle);
                None
            }
            Err(e) => Some(format!("Open failed: {e:#}")),
        };
        true
    }

    /// A scan or run in progress, which an opened session would mix with.
    fn busy(&self) -> bool {
        !self.twobody.idle() || self.manybody.is_busy()
    }

    /// Every tab as an opened session left it. A CLI scan has no inputs, so
    /// its settings stand in, and no structures, so Onebody's list stays.
    fn restore(&mut self, bundle: Bundle) {
        let Bundle {
            session,
            scan,
            run: record,
            trajectory,
        } = bundle;
        let Session {
            inputs,
            structures,
            scanned,
            estimate,
            run,
        } = session;
        let from_cli = inputs.is_none();
        self.tab = if scan.is_some() {
            Tab::Twobody
        } else {
            Tab::Onebody
        };
        let inputs = inputs.or_else(|| scan.as_ref().map(|s| s.settings.clone()));
        self.twobody.show_opened(scan, inputs.as_ref());
        if let Some(inputs) = &inputs {
            let molarity = run.as_ref().map(|r| r.molarity);
            self.conditions.restore(inputs, estimate, molarity);
            if let Some(warning) = self.forcefield.adopt(inputs) {
                self.twobody.warn(warning);
            }
        }
        self.manybody.restore(run.as_ref(), self.twobody.archive());
        let opened = record.map(|record| Opened { record, trajectory });
        self.manybody.show_opened(opened);
        if from_cli {
            return;
        }
        let ids = self.structures.replace(structures);
        // Indices were checked against the structures on load.
        if let Some([a, b]) = scanned {
            self.twobody
                .select_scanned([ids[a], ids[b]], &self.structures);
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
        if ui
            .add_enabled(!self.busy(), egui::Button::new("Open session…"))
            .on_hover_text(
                "Load a .duello.zip and restore every tab: structures, force field, \
                 conditions, scan inputs, Manybody inputs, and the scan and Manybody \
                 run if saved, replacing what is shown now. An opened run shows as \
                 finished; Run starts a new one",
            )
            .on_disabled_hover_text("Not while a scan or run is active")
            .clicked()
        {
            self.file.open();
        }
        if ui
            .button("Save session…")
            .on_hover_ui(|ui| {
                ui.label(self.save_tip());
            })
            .clicked()
        {
            self.save_session();
        }
        ui.separator();
        let has_scan = self.twobody.archive().is_some();
        ui.add_enabled_ui(has_scan, |ui| {
            ui.menu_button("Export", |ui| self.twobody.export_menu(ui))
                .response
                .on_hover_text("The shown scan's results as plain files")
                .on_disabled_hover_text("Run or open a scan first");
        });
    }

    /// What Save session writes now.
    fn save_tip(&self) -> String {
        let scan = match self.twobody.archive() {
            Some(_) => "the shown scan (table, CG bodies, PMF, poses)",
            None => "no scan yet",
        };
        let run = self
            .manybody
            .saved_results()
            .unwrap_or("no Manybody run yet");
        format!(
            "Every tab's inputs (structures, force field, conditions, scan and Manybody \
             inputs), the analysed molecule's beads and titration, {scan}, and {run}, as \
             .duello.zip. Keep it zipped: Safari's \"Open safe files after downloading\" \
             unzips downloads."
        )
    }

    fn ui_tab_strip(&mut self, ui: &mut egui::Ui) {
        ui.selectable_value(&mut self.tab, Tab::Onebody, "Onebody")
            .on_hover_text("Structures, force field and solution conditions shared by every tab");
        ui.selectable_value(&mut self.tab, Tab::Twobody, "Twobody")
            .on_hover_text("Orientation-averaged potential of mean force between two molecules");
        ui.selectable_value(&mut self.tab, Tab::Manybody, "Manybody")
            .on_hover_text("Monte Carlo of many copies of the Twobody scan's molecule");
    }

    /// Right to left: theme, GPU, then (natively) the version, which the
    /// browser shows in its HTML header instead.
    fn ui_status(&mut self, ui: &mut egui::Ui) {
        // Three small icons (System, Dark, Light), in reading order: the bar
        // lays out right to left.
        let size = egui::vec2(72.0, ui.available_height());
        let layout = egui::Layout::left_to_right(egui::Align::Center);
        ui.allocate_ui_with_layout(size, layout, egui::widgets::global_theme_preference_buttons);
        if self.twobody.gpu_available() {
            ui.weak("GPU ✓")
                .on_hover_text("A GPU/WebGPU adapter is available for scans");
        } else {
            ui.colored_label(egui::Color32::LIGHT_RED, "No GPU")
                .on_hover_text("Scans need WebGPU (Chrome 113+, Edge 113+)");
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::version::ui(ui);
        self.ui_notice(ui);
    }

    /// Leftmost on the right: the reason the last open or save failed.
    fn ui_notice(&mut self, ui: &mut egui::Ui) {
        let Some(notice) = &self.notice else { return };
        let label = egui::Label::new(egui::RichText::new(notice).color(egui::Color32::LIGHT_RED))
            .truncate()
            .sense(egui::Sense::click());
        if ui
            .add(label)
            .on_hover_text(format!("{notice}\n\nClick to dismiss"))
            .clicked()
        {
            self.notice = None;
        }
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
        if self.poll_open() {
            ctx.request_repaint();
        }
        // A busy many-body tab (running, GPU setup, late frames) needs frames.
        if self.manybody.is_busy() {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::{CgPolicy, Model};
    use crate::onebody::tests::tripeptide;
    use duello::archive::StructureFile;

    /// A session saved before any scan reopens with Onebody's structures
    /// and the inputs every tab showed.
    #[test]
    fn a_session_without_a_scan_restores_structures_and_inputs() {
        let mut saved = DuelloApp::blank();
        saved.structures.replace(vec![StructureFile {
            name: "tri.pdb".into(),
            contents: tripeptide(),
            ..Default::default()
        }]);
        (saved.conditions.ph, saved.conditions.ionic_strength) = (4.5, 0.02);
        (
            saved.conditions.temperature,
            saved.conditions.protein_millimolar,
        ) = (310.0, 2.0);
        saved.forcefield = ForceField {
            model: Model::KimHummer,
            cg: CgPolicy::Single,
        };
        let bytes = (saved.bundle(&saved.session(), None, None))
            .to_zip(&saved.extracts())
            .unwrap();

        let mut opened = DuelloApp::blank();
        opened.tab = Tab::Manybody;
        opened.restore(Bundle::from_zip(&bytes).unwrap());
        let names: Vec<_> = opened
            .structures
            .list()
            .map(|(_, n)| n.to_string())
            .collect();
        assert_eq!(names, ["tri.pdb"]);
        let c = &opened.conditions;
        assert_eq!((c.ph, c.ionic_strength, c.temperature), (4.5, 0.02, 310.0));
        assert_eq!(c.protein_millimolar, 2.0);
        assert_eq!(opened.forcefield, saved.forcefield);
        assert!(opened.twobody.archive().is_none());
        assert!(opened.tab == Tab::Onebody, "no scan to show");
    }
}
