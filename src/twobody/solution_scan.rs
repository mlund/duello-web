//! The Solution scan pane: what to vary, the run point by point, the B₂ curve
//! and its CSV. Results aren't saved in the session; the CSV is how they persist.

use super::job::{JobEvent, ScanJob};
use super::reference::Normaliser;
use super::summary::{b2_symbol, Key, Summary};
use super::sweep::{curve, Sweep, SweepPoints, Variable, POINTS};
use crate::transport::{Action, Phase, Transport};
use crate::widgets::{curve_color, curve_legend, grid, pick, placeholder, refit, row};
use duello::archive::StructureFile;
use duello::web_api::{GridSpec, ScanSettings};
use egui_plot::{HLine, HoverPosition, Line, LineStyle, Plot, Points};

/// Points run one after another; a point can't be held, only the sweep stopped.
const TRANSPORT: Transport = Transport {
    run_tip: "Run one full two-body scan per point, the others at Onebody's conditions",
    busy: Some("Sweeping…"),
    stop_tip: "Stop: abandon the running point; finished points stay",
};

/// What the plot's y axis shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Show {
    /// B₂/B₂ʰˢ, against the hard-sphere line at 1.
    Reduced,
    Absolute,
}

impl Show {
    const ALL: [Show; 2] = [Show::Reduced, Show::Absolute];

    fn label(self) -> &'static str {
        match self {
            Show::Reduced => "B₂/B₂ʰˢ",
            Show::Absolute => "B₂ (Å³)",
        }
    }

    /// The label for this pair, with `b2` B₂₂ or B₁₂.
    fn axis(self, b2: &str) -> String {
        match self {
            Show::Reduced => format!("{b2}/B₂ʰˢ"),
            Show::Absolute => format!("{b2} (Å³)"),
        }
    }

    fn key(self) -> Key {
        match self {
            Show::Reduced => Key::B2Star,
            Show::Absolute => Key::B2,
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Show::Reduced => "Reduced by B₂ʰˢ (set under Scan range); below 1: net attraction",
            Show::Absolute => "Absolute B₂, independent of the B₂ʰˢ choice",
        }
    }
}

/// What the pane asks of the Twobody tab, which owns the GPU and the downloads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SweepAction {
    Start,
    Stop,
    Save,
}

/// What a sweep needs from the Twobody panel when it starts.
pub struct Start {
    pub base: ScanSettings,
    pub grid: Option<GridSpec>,
    pub pair: [StructureFile; 2],
}

/// A started sweep: its points, and the running point's scan.
struct Run {
    points: SweepPoints,
    job: Option<ScanJob>,
    grid: Option<GridSpec>,
    pair: [StructureFile; 2],
    /// R-bins done of the running point.
    progress: f32,
}

pub struct SolutionScan {
    sweep: Sweep,
    show: Show,
    run: Option<Run>,
    error: Option<String>,
    /// Points and quantity last drawn; a change refits the plot.
    drawn: Option<(usize, Show)>,
}

impl Default for SolutionScan {
    fn default() -> Self {
        Self {
            sweep: Sweep::default(),
            show: Show::Reduced,
            run: None,
            error: None,
            drawn: None,
        }
    }
}

impl SolutionScan {
    pub fn running(&self) -> bool {
        self.run.as_ref().is_some_and(|r| r.job.is_some())
    }

    pub fn start(&mut self, start: Start) {
        self.error = None;
        let points = SweepPoints::new(self.sweep, start.base);
        let mut run = Run {
            points,
            job: None,
            grid: start.grid,
            pair: start.pair,
            progress: 0.0,
        };
        run.next();
        self.run = Some(run);
    }

    /// Finished points stay; the running point's job is handed back to retire.
    pub fn stop(&mut self) -> Option<ScanJob> {
        self.run.as_mut()?.job.take()
    }

    /// Records a finished point and starts the next. Called each frame.
    pub fn poll(&mut self) {
        let Some(run) = &mut self.run else { return };
        if let Err(e) = run.poll() {
            self.error = Some(format!("{e:#}"));
            run.job = None;
        }
    }

    pub fn has_points(&self) -> bool {
        self.run
            .as_ref()
            .is_some_and(|r| !r.points.points().is_empty())
    }

    /// None until a point is done.
    pub fn csv(&self, normaliser: &Normaliser) -> Option<String> {
        let run = self.run.as_ref()?;
        let [a, b] = run.pair.each_ref().map(|f| f.name.as_str());
        run.points.csv(normaliser, [a, b])
    }

    /// `blocked` says why a sweep can't start.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        normaliser: &Normaliser,
        blocked: Option<&str>,
    ) -> Option<SweepAction> {
        self.ui_controls(ui);
        let mut action = self.ui_transport(ui, blocked);
        if let Some(err) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, format!("Error: {err}"));
        }
        match &self.run {
            Some(run) if !run.points.points().is_empty() => {
                ui_show(ui, &mut self.show);
                let drawn = Some((run.points.points().len(), self.show));
                let reset = std::mem::replace(&mut self.drawn, drawn) != drawn;
                let summaries = run.points.summaries(normaliser);
                self.ui_plot(ui, &run.points, &summaries, reset);
                action = action.or(ui_caption(ui, &run.points, normaliser));
            }
            _ if self.running() => placeholder(ui, "The first point is running…"),
            _ => placeholder(ui, "Run a sweep to plot B₂ along it"),
        }
        action
    }

    fn ui_controls(&mut self, ui: &mut egui::Ui) {
        let idle = !self.running();
        ui.add_enabled_ui(idle, |ui| {
            grid("sweep_controls").show(ui, |ui| {
                self.ui_variable(ui);
                self.ui_range(ui);
                row(
                    ui,
                    "Points",
                    "Spaced linearly, both ends included; each is a full scan",
                    |ui| ui.add(egui::DragValue::new(&mut self.sweep.n).range(POINTS)),
                );
            });
        });
    }

    fn ui_variable(&mut self, ui: &mut egui::Ui) {
        let sweep = &mut self.sweep;
        let before = sweep.variable;
        row(ui, "Vary", sweep.variable.tip(), |ui| {
            pick(
                ui,
                "sweep_variable",
                &mut sweep.variable,
                &Variable::ALL,
                |v| v.label(),
                Some(Variable::tip),
            )
        });
        if sweep.variable != before {
            *sweep = Sweep::over(sweep.variable);
        }
    }

    fn ui_range(&mut self, ui: &mut egui::Ui) {
        let sweep = &mut self.sweep;
        let (bounds, unit) = (sweep.variable.bounds(), sweep.variable.unit());
        let tip = "First and last point; the other conditions are Onebody's";
        row(ui, "From", tip, |ui| {
            ui.horizontal(|ui| {
                let drag = |v| egui::DragValue::new(v).range(bounds.clone()).suffix(unit);
                ui.add(drag(&mut sweep.from)).on_hover_text(tip);
                ui.label("to").on_hover_text(tip);
                ui.add(drag(&mut sweep.to)).on_hover_text(tip);
            })
            .response
        });
    }

    fn ui_transport(&mut self, ui: &mut egui::Ui, blocked: Option<&str>) -> Option<SweepAction> {
        let phase = if self.running() {
            Phase::Running
        } else {
            Phase::Idle
        };
        let action = TRANSPORT.ui(ui, phase, blocked, None);
        match (&self.run, phase) {
            (Some(run), Phase::Running) => progress(ui, run),
            (_, Phase::Idle) => {
                if let Some(why) = blocked {
                    ui.weak(why);
                }
            }
            _ => {}
        }
        match action? {
            Action::Stop => Some(SweepAction::Stop),
            Action::Start => Some(SweepAction::Start),
            _ => None,
        }
    }

    /// `reset` refits the axes: a new point or another quantity may fall outside them.
    fn ui_plot(
        &self,
        ui: &mut egui::Ui,
        points: &SweepPoints,
        summaries: &[(f64, Summary)],
        mut reset: bool,
    ) {
        let curve = curve(summaries, self.show.key());
        let variable = points.sweep().variable;
        let b2 = b2_symbol(homodimer(points));
        let hover: Vec<String> = summaries.iter().map(|p| hover(variable, b2, p)).collect();
        let color = curve_color(ui.visuals().dark_mode);
        let y = self.show.axis(b2);
        let reduced = self.show == Show::Reduced;
        let plot = Plot::new("solution_scan_plot")
            .legend(curve_legend())
            .height((ui.available_height() - 24.0).max(120.0))
            .x_axis_label(variable.axis())
            .y_axis_label(y)
            .label_formatter(move |pos| match pos {
                HoverPosition::NearDataPoint { index, .. } => hover.get(*index).cloned(),
                _ => None,
            });
        refit(plot, &mut reset).show(ui, |plot| {
            if reduced {
                let hard_sphere =
                    HLine::new("hard sphere, 1", 1.0).style(LineStyle::dashed_dense());
                plot.hline(hard_sphere.width(1.5_f32));
            }
            plot.line(Line::new(b2, curve.clone()).color(color).width(2.0_f32));
            plot.points(Points::new(b2, curve).color(color).radius(4.0_f32));
        });
    }
}

/// The normaliser in use, and Save; the tab downloads, as File ▸ Export does.
fn ui_caption(
    ui: &mut egui::Ui,
    points: &SweepPoints,
    normaliser: &Normaliser,
) -> Option<SweepAction> {
    let reduced = points.reduced(normaliser)?;
    ui.horizontal(|ui| {
        ui.label(format!(
            "B₂ʰˢ = {:.3e} Å³ ({})",
            reduced.b2_hs, reduced.label
        ))
        .on_hover_text("The normaliser of B₂*, set under Scan range; display only");
        let save = ui.button("Save .csv").on_hover_text(CSV_TIP);
        save.clicked().then_some(SweepAction::Save)
    })
    .inner
}

/// Shared by the pane's button and File ▸ Export.
pub const CSV_TIP: &str = "Every point's conditions, thermodynamics and both molecules' \
     moments. The only copy: sweeps aren't saved in the session.";

impl Run {
    /// Starts the next point's scan, or ends the run when all are done.
    fn next(&mut self) {
        self.progress = 0.0;
        self.job = (self.points.next())
            .map(|(_, settings)| ScanJob::start(settings, self.grid, self.pair.clone()));
    }

    fn poll(&mut self) -> anyhow::Result<()> {
        let Some(job) = &self.job else { return Ok(()) };
        for event in job.poll() {
            match event {
                JobEvent::Progress(p) => self.progress = p.done as f32 / p.total.max(1) as f32,
                JobEvent::Done(done) => {
                    self.points.push(&*done?)?;
                    self.next();
                }
            }
        }
        Ok(())
    }
}

fn ui_show(ui: &mut egui::Ui, show: &mut Show) {
    ui.horizontal(|ui| {
        ui.label("Show").on_hover_text(show.tip());
        let tip = show.tip();
        pick(
            ui,
            "sweep_show",
            show,
            &Show::ALL,
            |s| s.label(),
            Some(Show::tip),
        )
        .on_hover_text(tip);
    });
}

fn progress(ui: &mut egui::Ui, run: &Run) {
    let (done, n) = (run.points.points().len(), run.points.sweep().n);
    let fraction = (done as f32 + run.progress) / n as f32;
    ui.add(egui::ProgressBar::new(fraction).text(format!("point {}/{n}", done + 1)))
        .on_hover_text("Points finished, and how far the running one has got");
}

/// Hover text of one point: x, B₂ and B₂*. Points without a B₂ are never
/// plotted, so hover indices match the curve's.
fn hover(variable: Variable, b2: &str, (x, s): &(f64, Summary)) -> String {
    let value = |k| s.value(k).unwrap_or(f64::NAN);
    format!(
        "{} = {x:.2}{}\n{b2} = {:.3e} Å³\n{b2}* = {:.2}",
        variable.label(),
        variable.unit(),
        value(Key::B2),
        value(Key::B2Star)
    )
}

/// From the first point; the request leaves it to the structures.
fn homodimer(points: &SweepPoints) -> bool {
    points.points().first().is_some_and(|p| p.facts.homodimer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finished point with no job, as after a sweep ends.
    fn finished() -> SolutionScan {
        let base = crate::test_archive::archive().settings;
        let mut points = SweepPoints::new(Sweep::default(), base);
        points.push(&crate::test_archive::associating()).unwrap();
        let file = StructureFile {
            name: "a.pdb".into(),
            contents: Vec::new(),
            chains: Vec::new(),
        };
        SolutionScan {
            run: Some(Run {
                points,
                job: None,
                grid: None,
                pair: [file.clone(), file],
                progress: 0.0,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn the_pane_renders_before_and_after_a_sweep() {
        let steric = Normaliser::Steric;
        crate::widgets::headless(|ui| {
            SolutionScan::default().ui(ui, &steric, Some("why"));
        });
        let mut scan = finished();
        for show in Show::ALL {
            scan.show = show;
            crate::widgets::headless(|ui| {
                scan.ui(ui, &steric, None);
            });
        }
        assert!(scan
            .csv(&steric)
            .is_some_and(|csv| csv.contains("a.pdb/a.pdb")));
    }
}
