//! The Solution scan pane: what to vary, the run point by point, the B₂ curve
//! and its CSV. Results aren't saved in the session; the CSV is how they persist.

use super::job::{JobEvent, ScanJob};
use super::reference::Normaliser;
use super::summary::{b2_symbol, Key, Summary};
use super::sweep::{curve, vant_hoff, Sweep, SweepPoints, Variable, POINTS};
use super::vant_hoff::VantHoff;
use crate::transport::{Action, Phase, Transport};
use crate::widgets::{curve_color, curve_legend, grid, log_scaled, pick, placeholder, refit, row};
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
    /// log₁₀ K_d; only where the pair attracts beyond excluded volume.
    Kd,
    /// ln K_d against 1000/T with the fit; temperature sweeps only.
    VantHoff,
}

impl Show {
    const ALL: [Show; 4] = [Show::Reduced, Show::Absolute, Show::Kd, Show::VantHoff];

    fn label(self) -> &'static str {
        match self {
            Show::Reduced => "B₂/B₂ʰˢ",
            Show::Absolute => "B₂ (Å³)",
            Show::Kd => "K_d",
            Show::VantHoff => "van 't Hoff",
        }
    }

    /// The label for this pair, with `b2` B₂₂ or B₁₂.
    fn axis(self, b2: &str) -> String {
        match self {
            Show::Reduced => format!("{b2}/B₂ʰˢ"),
            Show::Absolute => format!("{b2} (Å³)"),
            Show::Kd => "log₁₀ K_d (M)".into(),
            Show::VantHoff => "ln K_d (K_d in M)".into(),
        }
    }

    fn key(self) -> Key {
        match self {
            Show::Reduced => Key::B2Star,
            Show::Absolute => Key::B2,
            Show::Kd | Show::VantHoff => Key::Kd,
        }
    }

    /// Plot coordinates of `value` at sweep point `x`; None off a log scale.
    fn at(self, x: f64, value: f64) -> Option<[f64; 2]> {
        match self {
            Show::Reduced | Show::Absolute => Some([x, value]),
            Show::Kd => Some([x, log_scaled(value, true)?]),
            Show::VantHoff => (value > 0.0).then(|| [1000.0 / x, value.ln()]),
        }
    }

    /// Legend name of the plotted points.
    fn series(self, b2: &str) -> &str {
        match self {
            Show::Reduced | Show::Absolute => b2,
            Show::Kd => "K_d",
            Show::VantHoff => "ln K_d",
        }
    }

    fn x_axis(self, variable: Variable) -> &'static str {
        match self {
            Show::VantHoff => "1000/T (K⁻¹)",
            _ => variable.axis(),
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Show::Reduced => "Reduced by B₂ʰˢ (set under Scan range); below 1: net attraction",
            Show::Absolute => "Absolute B₂, independent of the B₂ʰˢ choice",
            Show::Kd => {
                "Dissociation constant 1/K_a with K_a = −2(B₂ − B₂ʳᵉᶠ), one B₂ʳᵉᶠ for \
                 every point; points without attraction beyond it have none"
            }
            Show::VantHoff => {
                "ln K_d against 1000/T, with the fit of constant ΔC°_p dashed. Apparent: \
                 the model's potentials depend on T through ε_r and the Debye length, \
                 so ΔH° is the model's van 't Hoff enthalpy, not a calorimetric one"
            }
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
                let summaries = run.points.summaries(normaliser);
                let kd = curve(&summaries, Key::Kd);
                let variable = run.points.sweep().variable;
                let fit = vant_hoff(variable, &kd);
                let show = ui_show(ui, &mut self.show, &offered(&kd, fit.is_some()));
                let drawn = Some((run.points.points().len(), show));
                let reset = std::mem::replace(&mut self.drawn, drawn) != drawn;
                let fit = fit.filter(|_| show == Show::VantHoff);
                let fitted = fit.map(|fit| fit_curve(&fit, &kd));
                ui_plot(ui, show, &run.points, &summaries, fitted, reset);
                action = action.or(ui_caption(ui, &run.points, normaliser, fit.as_ref()));
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
}

/// `reset` refits the axes: a new point or another quantity may fall outside them.
fn ui_plot(
    ui: &mut egui::Ui,
    show: Show,
    points: &SweepPoints,
    summaries: &[(f64, Summary)],
    fitted: Option<Vec<[f64; 2]>>,
    mut reset: bool,
) {
    let key = show.key();
    let variable = points.sweep().variable;
    let b2 = b2_symbol(homodimer(points));
    // One pass, so hover indices match the plotted points.
    let (curve, hover): (Vec<[f64; 2]>, Vec<String>) = (summaries.iter())
        .filter_map(|p| Some((show.at(p.0, p.1.value(key)?)?, hover(variable, b2, p))))
        .unzip();
    let color = curve_color(ui.visuals().dark_mode);
    let y = show.axis(b2);
    let reduced = show == Show::Reduced;
    let plot = Plot::new("solution_scan_plot")
        .legend(curve_legend())
        .height((ui.available_height() - 24.0).max(120.0))
        .x_axis_label(show.x_axis(variable))
        .y_axis_label(y)
        .label_formatter(move |pos| match pos {
            HoverPosition::NearDataPoint { index, .. } => hover.get(*index).cloned(),
            _ => None,
        });
    refit(plot, &mut reset).show(ui, |plot| {
        if reduced {
            let hard_sphere = HLine::new("hard sphere, 1", 1.0).style(LineStyle::dashed_dense());
            plot.hline(hard_sphere.width(1.5_f32));
        }
        let name = show.series(b2);
        plot.line(Line::new(name, curve.clone()).color(color).width(2.0_f32));
        plot.points(Points::new(name, curve).color(color).radius(4.0_f32));
        if let Some(fitted) = fitted {
            // Its samples aren't sweep points, so hovering them would show another point's text.
            let fit = Line::new("fit, constant ΔC°_p", fitted)
                .style(LineStyle::dashed_loose())
                .allow_hover(false);
            plot.line(fit.color(color).width(1.5_f32));
        }
    });
}

/// The normaliser in use, and Save; the tab downloads, as File ▸ Export does.
fn ui_caption(
    ui: &mut egui::Ui,
    points: &SweepPoints,
    normaliser: &Normaliser,
    fit: Option<&VantHoff>,
) -> Option<SweepAction> {
    let reduced = points.reduced(normaliser)?;
    if let Some(fit) = fit {
        ui.label(fit.caption()).on_hover_text(FIT_TIP);
    }
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

const FIT_TIP: &str = "Least squares of ln K_d with ΔH°_d and ΔS°_d drifting through a \
     constant ΔC°_p,d, reported at the mean T swept; dissociation, 1 M standard state. \
     ± is the standard error from the scatter about the fit, not from each point's K_d; \
     a narrow T range leaves ΔC°_p poorly determined. Association values are the \
     negatives. Apparent: the potentials depend on T";

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

/// The fit across the swept temperatures, on the van 't Hoff axes.
fn fit_curve(fit: &VantHoff, kd: &[[f64; 2]]) -> Vec<[f64; 2]> {
    const SAMPLES: usize = 50;
    let (low, high) = kd.iter().fold((f64::INFINITY, 0.0_f64), |(l, h), [t, _]| {
        (l.min(*t), h.max(*t))
    });
    (0..SAMPLES)
        .map(|i| low + (high - low) * i as f64 / (SAMPLES - 1) as f64)
        .filter_map(|t| Show::VantHoff.at(t, fit.line(t).exp()))
        .collect()
}

/// What the picker lists, given `kd` as `[x, K_d]` and whether it has a fit.
fn offered(kd: &[[f64; 2]], fits: bool) -> Vec<Show> {
    Show::ALL
        .into_iter()
        .filter(|show| match show {
            Show::Reduced | Show::Absolute => true,
            Show::Kd => !kd.is_empty(),
            Show::VantHoff => fits,
        })
        .collect()
}

/// The view drawn: the chosen one, or B₂/B₂ʰˢ while the sweep can't show it,
/// e.g. van 't Hoff before four temperatures are in. The choice itself stays.
fn ui_show(ui: &mut egui::Ui, chosen: &mut Show, options: &[Show]) -> Show {
    let mut show = if options.contains(chosen) {
        *chosen
    } else {
        Show::Reduced
    };
    let before = show;
    ui.horizontal(|ui| {
        ui.label("Show").on_hover_text(show.tip());
        let tip = show.tip();
        let id = "sweep_show";
        pick(ui, id, &mut show, options, |s| s.label(), Some(Show::tip)).on_hover_text(tip);
    });
    if show != before {
        *chosen = show;
    }
    show
}

fn progress(ui: &mut egui::Ui, run: &Run) {
    let (done, n) = (run.points.points().len(), run.points.sweep().n);
    let fraction = (done as f32 + run.progress) / n as f32;
    ui.add(egui::ProgressBar::new(fraction).text(format!("point {}/{n}", done + 1)))
        .on_hover_text("Points finished, and how far the running one has got");
}

/// Hover text of one point: x, B₂, B₂* and K_d where the pair has one.
fn hover(variable: Variable, b2: &str, (x, s): &(f64, Summary)) -> String {
    let value = |k| s.value(k).unwrap_or(f64::NAN);
    let kd = (s.value(Key::Kd)).map_or(String::new(), |kd| format!("\nK_d = {kd:.3e} M"));
    format!(
        "{} = {x:.2}{}\n{b2} = {:.3e} Å³\n{b2}* = {:.2}{kd}",
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
    use super::super::vant_hoff::kd_by_t;
    use super::*;

    /// A finished point with no job, as after a sweep ends.
    fn finished() -> SolutionScan {
        finished_along(Sweep::default(), 1)
    }

    /// `n` finished points of an associating pair along `sweep`.
    fn finished_along(sweep: Sweep, n: usize) -> SolutionScan {
        let base = crate::test_archive::archive().settings;
        let mut points = SweepPoints::new(sweep, base);
        for _ in 0..n {
            points.push(&crate::test_archive::associating()).unwrap();
        }
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
    fn k_d_is_offered_with_values_and_van_t_hoff_only_along_t() {
        use Variable::*;
        let options = |v, kd: &[[f64; 2]]| offered(kd, vant_hoff(v, kd).is_some());
        assert_eq!(options(Salt, &[]), [Show::Reduced, Show::Absolute]);
        assert_eq!(
            options(Salt, &kd_by_t()),
            [Show::Reduced, Show::Absolute, Show::Kd]
        );
        assert_eq!(options(Temperature, &kd_by_t()), Show::ALL);
        assert!(!options(Temperature, &kd_by_t()[..3]).contains(&Show::VantHoff));
    }

    #[test]
    fn k_d_plots_on_log_scales() {
        assert_eq!(Show::Kd.at(300.0, 1e-6), Some([300.0, -6.0]));
        let [x, y] = Show::VantHoff.at(250.0, 1e-6).unwrap();
        assert_eq!(x, 4.0);
        assert!((y - 1e-6_f64.ln()).abs() < 1e-12);
        assert_eq!(Show::Absolute.at(300.0, -5.0), Some([300.0, -5.0]));
        assert_eq!(Show::Kd.at(300.0, 0.0), None, "no logarithm");
    }

    /// The fit spans the swept temperatures, on the van 't Hoff axes.
    #[test]
    fn the_fitted_curve_spans_the_sweep() {
        let fit = VantHoff::fit(&kd_by_t()).unwrap();
        let line = fit_curve(&fit, &kd_by_t());
        let (first, last) = (line[0], *line.last().unwrap());
        assert_eq!([first[0], last[0]], [1000.0 / 280.0, 1000.0 / 310.0]);
        assert!((first[1] - fit.line(280.0)).abs() < 1e-12);
    }

    /// Four temperatures reach the K_d and van 't Hoff views, and the CSV fit.
    #[test]
    fn a_temperature_sweep_renders_every_view() {
        let sweep = Sweep {
            variable: Variable::Temperature,
            from: 280.0,
            to: 310.0,
            n: 4,
        };
        let mut scan = finished_along(sweep, 4);
        let steric = Normaliser::Steric;
        for show in Show::ALL {
            scan.show = show;
            crate::widgets::headless(|ui| {
                scan.ui(ui, &steric, None);
            });
            assert_eq!(scan.drawn.map(|d| d.1), Some(show), "{show:?} was offered");
        }
        assert!(scan
            .csv(&steric)
            .is_some_and(|csv| csv.contains("# van 't Hoff")));
    }

    /// One temperature can't be fitted yet: draw B₂/B₂ʰˢ, keep the choice.
    #[test]
    fn an_unavailable_view_falls_back_but_stays_chosen() {
        let sweep = Sweep {
            variable: Variable::Temperature,
            ..Sweep::default()
        };
        let mut scan = finished_along(sweep, 1);
        scan.show = Show::VantHoff;
        crate::widgets::headless(|ui| {
            scan.ui(ui, &Normaliser::Steric, None);
        });
        assert_eq!(scan.drawn.map(|d| d.1), Some(Show::Reduced));
        assert_eq!(scan.show, Show::VantHoff);
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
