//! Manybody plots: the energy trace, the scattering curves and g(r). Views,
//! log scaling and error bars stay in here.

use duello::nbody::{RdfCurve, ScatteringCurve};
use egui_plot::{Line, MarkerShape, Plot, PlotPoints, Points};

#[derive(Clone, Copy, PartialEq, Debug)]
enum View {
    StructureFactor,
    Intensity,
    FormFactor,
}

/// How a series is drawn: simulated curves solid, the theory dashed.
#[derive(Clone, Copy)]
enum Style {
    Primary,
    Secondary,
    Reference,
}

struct Series<'a> {
    name: &'static str,
    values: &'a [f64],
    /// Where copies give them.
    errors: Option<&'a [f64]>,
    style: Style,
    /// Symbols at each point; off for dense curves such as g(r).
    markers: bool,
}

impl<'a> Series<'a> {
    fn new(name: &'static str, values: &'a [f64], errors: Option<&'a [f64]>, style: Style) -> Self {
        Self {
            name,
            values,
            errors,
            style,
            markers: true,
        }
    }

    fn without_markers(self) -> Self {
        Self {
            markers: false,
            ..self
        }
    }
}

impl View {
    const ALL: [View; 3] = [View::StructureFactor, View::Intensity, View::FormFactor];

    fn label(self) -> &'static str {
        match self {
            View::StructureFactor => "S(q)",
            View::Intensity => "I(q)",
            View::FormFactor => "P(q)",
        }
    }

    /// I and P are per molecule, i.e. I/c up to contrast and molar mass.
    fn axis_label(self) -> &'static str {
        match self {
            View::StructureFactor => "S(q)",
            View::Intensity => "I(q)/c (relative)",
            View::FormFactor => "P(q) (relative)",
        }
    }

    /// I and P come from the bead sums, which only detailed runs do.
    fn offered(self, detailed: bool) -> bool {
        detailed || self == View::StructureFactor
    }

    /// S_com and its theory are cheap, so every run shows them; S_eff joins
    /// when beads are summed.
    fn curves(self, curve: &ScatteringCurve, detailed: bool) -> Vec<Series<'_>> {
        match self {
            View::StructureFactor => {
                let mut all = vec![Series::new(
                    "S_com",
                    &curve.s_com,
                    curve.s_com_error.as_deref(),
                    Style::Primary,
                )];
                if detailed {
                    all.push(Series::new(
                        "S_eff",
                        &curve.s_eff,
                        curve.s_eff_error.as_deref(),
                        Style::Secondary,
                    ));
                }
                if !curve.s_hsy.is_empty() {
                    all.push(Series::new(
                        "charged sphere (MPB-RMSA)",
                        &curve.s_hsy,
                        None,
                        Style::Reference,
                    ));
                }
                all
            }
            View::Intensity => vec![Series::new("I(q)", &curve.intensity, None, Style::Primary)],
            View::FormFactor => vec![Series::new(
                "P(q)",
                &curve.form_factor,
                None,
                Style::Primary,
            )],
        }
    }
}

#[derive(Clone, Copy)]
struct LogAxes {
    x: bool,
    y: bool,
}

/// Which scattering curve is shown, and each one's axis scales.
pub struct ScatteringView {
    view: View,
    /// Indexed by `View as usize`.
    log_axes: [LogAxes; View::ALL.len()],
}

impl Default for ScatteringView {
    fn default() -> Self {
        Self {
            view: View::StructureFactor,
            // Log-log spreads the low-q rise, where S and I differ most.
            log_axes: [LogAxes { x: true, y: true }; View::ALL.len()],
        }
    }
}

/// `detailed` runs also offer the bead-level I(q) and P(q). Plots take the
/// run's id so each run starts fitted to its own data.
pub fn scattering(
    ui: &mut egui::Ui,
    curve: &ScatteringCurve,
    state: &mut ScatteringView,
    detailed: bool,
    run_id: u64,
) {
    if !state.view.offered(detailed) {
        state.view = View::StructureFactor;
    }
    scattering_controls(ui, state, detailed);
    let view = state.view;
    let axes = state.log_axes[view as usize];
    let all = view.curves(curve, detailed);
    let dark = ui.visuals().dark_mode;
    // One id per view and scale, so each fits its own bounds when shown.
    // Also per series set, so S_eff appearing doesn't inherit stale bounds.
    Plot::new((
        "manybody_plot",
        run_id,
        view.label(),
        axes.x,
        axes.y,
        detailed,
    ))
    .legend(curve_legend())
    .x_axis_label(log_label("q (Å⁻¹)", axes.x))
    .y_axis_label(log_label(view.axis_label(), axes.y))
    .show(ui, |plot| {
        for s in &all {
            draw_series(plot, &curve.q, s, axes, dark);
        }
    });
}

fn scattering_controls(ui: &mut egui::Ui, state: &mut ScatteringView, detailed: bool) {
    let axes = &mut state.log_axes[state.view as usize];
    ui.horizontal(|ui| {
        egui::ComboBox::from_label("Plot")
            .selected_text(state.view.label())
            .show_ui(ui, |ui| {
                for view in View::ALL.into_iter().filter(|v| v.offered(detailed)) {
                    ui.selectable_value(&mut state.view, view, view.label());
                }
            })
            .response
            .on_hover_text("S(q) for every run; I(q) and P(q) with Detailed I(q)");
        ui.checkbox(&mut axes.x, "log q")
            .on_hover_text("Logarithmic q axis");
        ui.checkbox(&mut axes.y, "log y")
            .on_hover_text("Logarithmic y axis; non-positive points are hidden");
    });
}

/// g(r) with copy error bars against the uncorrelated g = 1. Linear axes:
/// g is order one and its zeros (overlap) are part of the story.
pub fn pair_correlation(ui: &mut egui::Ui, curve: &RdfCurve, run_id: u64) {
    let g = Series::new("g(r)", &curve.g, curve.error.as_deref(), Style::Primary).without_markers();
    let linear = LogAxes { x: false, y: false };
    let dark = ui.visuals().dark_mode;
    let ideal = egui_plot::HLine::new("uncorrelated, g = 1", 1.0)
        .color(style_color(Style::Reference, dark))
        .width(2.0_f32)
        .style(egui_plot::LineStyle::dashed_dense());
    Plot::new(("pair_correlation", run_id))
        .legend(curve_legend())
        .x_axis_label("r (Å)")
        .y_axis_label("g(r)")
        .show(ui, |plot| {
            draw_series(plot, &curve.r, &g, linear, dark);
            plot.hline(ideal);
        });
}

/// (sweep, U/N) points. Follows the recent energy range until the user
/// drags or zooms; double-click resumes following. A new `key` (run and
/// trace epoch) gives a fresh view.
pub fn energy(ui: &mut egui::Ui, trace: &[[f64; 2]], follow: &mut bool, key: (u64, u64)) {
    let color = curve_color(ui.visuals().dark_mode);
    let following = *follow;
    let response = Plot::new(("energy_trace", key))
        .legend(egui_plot::Legend::default())
        .x_axis_label("Sweeps")
        .y_axis_label("U / N (kT)")
        .show(ui, |plot| {
            marked_line(plot, "U/N", trace.to_vec(), color);
            if let Some((lo, hi)) = recent_range(trace).filter(|_| following) {
                plot.set_plot_bounds_y(lo..=hi);
            }
        })
        .response;
    let zoomed = response.hovered()
        && ui.input(|i| i.zoom_delta() != 1.0 || i.smooth_scroll_delta != egui::Vec2::ZERO);
    if response.double_clicked() {
        *follow = true;
    } else if response.dragged() || zoomed {
        *follow = false;
    }
}

/// egui_plot has no log axes, so log scales plot log₁₀ of the data;
/// non-positive values have no logarithm and are dropped.
fn scaled(v: f64, log: bool) -> Option<f64> {
    match log {
        false => Some(v),
        true => (v > 0.0).then(|| v.log10()),
    }
}

fn point(q: f64, v: f64, axes: LogAxes) -> Option<[f64; 2]> {
    Some([scaled(q, axes.x)?, scaled(v, axes.y)?])
}

fn to_points(q: &[f64], values: &[f64], axes: LogAxes) -> Vec<[f64; 2]> {
    q.iter()
        .zip(values)
        .filter_map(|(&q, &v)| point(q, v, axes))
        .collect()
}

fn log_label(label: &str, log: bool) -> String {
    if log {
        format!("log₁₀ {label}")
    } else {
        label.to_string()
    }
}

/// The Twobody tab's PMF blue, so both tabs read alike.
fn curve_color(dark_mode: bool) -> egui::Color32 {
    if dark_mode {
        egui::Color32::from_rgb(100, 150, 255)
    } else {
        egui::Color32::from_rgb(30, 60, 180)
    }
}

fn style_color(style: Style, dark_mode: bool) -> egui::Color32 {
    let rgb = |dark: (u8, u8, u8), light: (u8, u8, u8)| {
        let (r, g, b) = if dark_mode { dark } else { light };
        egui::Color32::from_rgb(r, g, b)
    };
    match style {
        Style::Primary => curve_color(dark_mode),
        Style::Secondary => rgb((90, 210, 140), (0, 130, 70)),
        Style::Reference => rgb((255, 170, 80), (200, 90, 0)),
    }
}

fn draw_series(
    plot: &mut egui_plot::PlotUi,
    q: &[f64],
    s: &Series,
    axes: LogAxes,
    dark_mode: bool,
) {
    let color = style_color(s.style, dark_mode);
    let points = to_points(q, s.values, axes);
    if let Style::Reference = s.style {
        plot.line(
            Line::new(s.name, PlotPoints::from(points))
                .color(color)
                .width(2.0_f32)
                .style(egui_plot::LineStyle::dashed_dense()),
        );
        return;
    }
    for bar in s
        .errors
        .map(|e| error_bars(q, s.values, e, axes))
        .into_iter()
        .flatten()
    {
        plot.line(Line::new(s.name, PlotPoints::from(bar.to_vec())).color(color));
    }
    if s.markers {
        marked_line(plot, s.name, points, color);
    } else {
        plot.line(
            Line::new(s.name, PlotPoints::from(points))
                .color(color)
                .width(3.0_f32),
        );
    }
}

/// Lower right, clear of the low-q rise and g(r)'s contact peak on the left.
fn curve_legend() -> egui_plot::Legend {
    egui_plot::Legend::default().position(egui_plot::Corner::RightBottom)
}

/// U-range over the last half of the sweeps, where the run still drifts
/// slowly; the early drop would otherwise flatten it.
fn recent_range(trace: &[[f64; 2]]) -> Option<(f64, f64)> {
    let half = trace.last()?[0] / 2.0;
    let (lo, hi) = trace
        .iter()
        .filter(|[sweep, _]| *sweep >= half)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), [_, u]| {
            (lo.min(*u), hi.max(*u))
        });
    // A flat trace still needs a visible span.
    let pad = 0.05 * (hi - lo).max(1e-3 * lo.abs().max(1.0));
    Some((lo - pad, hi + pad))
}

/// Twobody style: a line through symbols marking the sampled points.
fn marked_line(
    plot: &mut egui_plot::PlotUi,
    name: &str,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
) {
    plot.line(
        Line::new(name, PlotPoints::from(points.clone()))
            .color(color)
            .width(3.0_f32),
    );
    // Same name, so one legend entry toggles line and markers together.
    plot.points(
        Points::new(name, PlotPoints::from(points))
            .color(color)
            .shape(MarkerShape::Circle)
            .radius(4.0_f32),
    );
}

/// Vertical ±σ bars, one per q. On log y, a bar reaching zero is cut one
/// decade below its value: dropping it would make noisy points look certain.
fn error_bars(q: &[f64], values: &[f64], errors: &[f64], axes: LogAxes) -> Vec<[[f64; 2]; 2]> {
    (0..q.len())
        .filter_map(|k| {
            let (v, e) = (values[k], errors[k]);
            let low = if axes.y && v - e <= 0.0 {
                v / 10.0
            } else {
                v - e
            };
            Some([point(q[k], low, axes)?, point(q[k], v + e, axes)?])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A large initial drop must not set the scale: only the last half of
    /// the sweeps counts, padded by 5% of its span.
    #[test]
    fn recent_range_ignores_the_initial_drop() {
        let trace = [
            [0.0, 100.0],
            [10.0, 10.0],
            [20.0, -1.0],
            [30.0, -2.0],
            [40.0, -3.0],
        ];
        let (lo, hi) = recent_range(&trace).unwrap();
        // Sweeps ≥ 20: U from −3 to −1, span 2, pad 0.1.
        assert!(
            (lo + 3.1).abs() < 1e-12 && (hi + 0.9).abs() < 1e-12,
            "{lo} {hi}"
        );
    }

    fn curve() -> ScatteringCurve {
        ScatteringCurve {
            q: vec![0.1],
            intensity: vec![1.0],
            form_factor: vec![2.0],
            s_eff: vec![3.0],
            s_eff_error: None,
            s_hsy: vec![4.0],
            s_com: vec![5.0],
            s_com_error: None,
            frames: 1,
        }
    }

    fn names(view: View, detailed: bool) -> Vec<&'static str> {
        let curve = curve();
        view.curves(&curve, detailed)
            .iter()
            .map(|s| s.name)
            .collect()
    }

    #[test]
    fn structure_factor_always_shows_mass_centres_and_reference() {
        assert_eq!(
            names(View::StructureFactor, false),
            ["S_com", "charged sphere (MPB-RMSA)"]
        );
        assert_eq!(
            names(View::StructureFactor, true),
            ["S_com", "S_eff", "charged sphere (MPB-RMSA)"]
        );
    }

    #[test]
    fn bead_views_need_a_detailed_run() {
        let offered = |detailed| {
            View::ALL
                .into_iter()
                .filter(|v| v.offered(detailed))
                .collect::<Vec<_>>()
        };
        assert_eq!(offered(false), [View::StructureFactor]);
        assert_eq!(offered(true), View::ALL);
        assert_eq!(names(View::Intensity, true), ["I(q)"]);
    }

    /// S = 0.1 ± 0.15 must still show its bar on log y, cut one decade down.
    #[test]
    fn log_error_bars_reaching_below_zero_are_clipped_not_dropped() {
        let axes = LogAxes { x: false, y: true };
        let bars = error_bars(&[0.2], &[0.1], &[0.15], axes);
        assert_eq!(bars.len(), 1);
        let [[_, low], [_, high]] = bars[0];
        assert!((low + 2.0).abs() < 1e-12, "{low}");
        assert!((high - 0.25_f64.log10()).abs() < 1e-12, "{high}");
    }

    #[test]
    fn recent_range_of_a_flat_trace_is_not_empty() {
        let (lo, hi) = recent_range(&[[0.0, 5.0], [10.0, 5.0]]).unwrap();
        assert!(lo < 5.0 && hi > 5.0);
        assert!(recent_range(&[]).is_none());
    }
}
