//! The Free energy window: curves against R, streamed while a scan runs.

use crate::widgets::{curve_color, curve_legend, pick, refit};
use duello::archive::ScanArchive;
use duello::web_api::PmfPoint;
use duello::VirialCoeff;
use egui_plot::{Line, MarkerShape, Plot, PlotBounds, PlotPoint, Points};

/// Which curve the window plots.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Shown {
    #[default]
    FreeEnergy,
    MeanForce,
    HeatCapacity,
    VirialRunning,
    B2Start,
}

impl Shown {
    const ALL: [Shown; 5] = [
        Shown::FreeEnergy,
        Shown::MeanForce,
        Shown::HeatCapacity,
        Shown::VirialRunning,
        Shown::B2Start,
    ];

    fn label(&self) -> &'static str {
        match self {
            Shown::FreeEnergy => "Free energy",
            Shown::MeanForce => "Mean force",
            Shown::HeatCapacity => "Heat capacity",
            Shown::VirialRunning => "B₂₂ up to R",
            Shown::B2Start => "B₂₂ by start",
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Shown::FreeEnergy => {
                "Potential of mean force w(R), the orientational mean energy ⟨U⟩(R) \
                 and the entropic part −TS(R) = w − ⟨U⟩, at fixed pair potential"
            }
            Shown::MeanForce => {
                "−dw/dR, the angle-averaged force between the mass centres; \
                 positive pushes apart"
            }
            Shown::HeatCapacity => {
                "Orientational heat capacity C(R)/k = ⟨δU²⟩/(kT)², the energy spread \
                 over orientations at R; peaks where orientations compete"
            }
            Shown::VirialRunning => {
                "B₂₂ integrated out to R; it should level off well before the last R, \
                 else extend the scan range. Shown once the scan finishes"
            }
            Shown::B2Start => {
                "B₂₂ as if the scan had started at R, with hard spheres below; it should \
                 level off towards the smallest R, else lower R_min. Shown once the scan finishes"
            }
        }
    }

    /// B₂₂ by start is read at small R, so the wall stays in view.
    fn is_cut(self) -> bool {
        self != Shown::B2Start
    }

    /// The tip, plus where the axes start for views that cut them.
    fn full_tip(self) -> String {
        if !self.is_cut() {
            return self.tip().into();
        }
        format!(
            "{}\n\nOnce the scan ends, axes start where w(R) < {WALL} kT; double-click to show all",
            self.tip()
        )
    }

    fn y_label(self) -> &'static str {
        match self {
            Shown::FreeEnergy => "Energy (kT)",
            Shown::MeanForce => "Mean force (kT/Å)",
            Shown::HeatCapacity => "C(R)/k",
            Shown::VirialRunning | Shown::B2Start => "B₂₂ (Å³)",
        }
    }
}

/// w(R) in kT above which the axes leave R out: contact walls reach hundreds of kT.
const WALL: f64 = 10.0;

/// A series' colour for the theme, dark or not.
type Colour = fn(bool) -> egui::Color32;

fn red(dark: bool) -> egui::Color32 {
    let [r, g, b] = if dark { [255, 120, 100] } else { [200, 40, 30] };
    egui::Color32::from_rgb(r, g, b)
}

fn green(dark: bool) -> egui::Color32 {
    let [r, g, b] = if dark { [110, 200, 120] } else { [30, 130, 50] };
    egui::Color32::from_rgb(r, g, b)
}

/// (name, data, colour, marker) of one plotted series.
type Series<'a> = (&'static str, &'a [PlotPoint], Colour, MarkerShape);

/// The series the window holds, filled as R-bins arrive.
#[derive(Default)]
pub struct Curves {
    free_energy: Vec<PlotPoint>,
    mean_energy: Vec<PlotPoint>,
    /// −TS = w − ⟨U⟩, the entropic part of the free energy.
    entropy: Vec<PlotPoint>,
    mean_force: Vec<PlotPoint>,
    heat_capacity: Vec<PlotPoint>,
    /// B₂₂ out to R; needs the whole PMF, so only a finished scan has it.
    b2: Vec<PlotPoint>,
    /// B₂₂ by scan start R′; likewise only once finished.
    b2_start: Vec<PlotPoint>,
    /// The whole PMF is in; only then is the wall cut final.
    finished: bool,
    /// Refit the axes next frame: a scan started or ended, or another curve shows.
    reset: bool,
}

impl Curves {
    /// Empty, for a scan about to stream in.
    pub fn new() -> Self {
        Self {
            reset: true,
            ..Default::default()
        }
    }

    pub fn push(&mut self, p: PmfPoint) {
        let r = f64::from(p.r);
        let (w, u) = (f64::from(p.free_energy), f64::from(p.mean_energy));
        self.free_energy.push(PlotPoint::new(r, w));
        self.mean_energy.push(PlotPoint::new(r, u));
        self.entropy.push(PlotPoint::new(r, w - u));
        self.heat_capacity
            .push(PlotPoint::new(r, f64::from(p.energy_variance)));
        if let [.., a, b, c] = self.free_energy[..] {
            self.mean_force.extend(mean_force(&[a, b, c]));
        }
    }

    /// Replaces the streamed curves with the finished scan's, which also gives B₂₂(R).
    pub fn finish(&mut self, archive: &ScanArchive) {
        *self = Self::new();
        archive.points.iter().for_each(|&p| self.push(p));
        let sigma = archive.virial.sigma();
        self.b2 = plot_points(VirialCoeff::running(&archive.points, sigma));
        self.b2_start = plot_points(VirialCoeff::by_start(&archive.points, sigma));
        self.finished = true;
    }

    /// A picker over the curves, then the picked one's plot.
    pub fn ui(&mut self, ui: &mut egui::Ui, shown: &mut Shown, is_dark: bool) {
        let before = *shown;
        ui.horizontal(|ui| {
            let tip = shown.full_tip();
            ui.label("Show").on_hover_text(&tip);
            pick(
                ui,
                "twobody_curve",
                shown,
                &Shown::ALL,
                |s| s.label(),
                Some(Shown::tip),
            )
            .on_hover_text(tip);
        });
        // Other units, so the old axes would hide the new curve.
        self.reset |= *shown != before;
        let plot = Plot::new("pmf_plot")
            .legend(curve_legend())
            .x_axis_label("Mass-centre separation, R (Å)")
            .y_axis_label(shown.y_label());
        // Only on reset, so the user's pan and zoom stand; read it before
        // `refit` clears it.
        let fit = self.reset.then(|| self.bounds(*shown)).flatten();
        let plot = refit(plot, &mut self.reset);
        let series = self.series(*shown);
        plot.show(ui, |plot_ui| {
            if let Some(bounds) = fit {
                plot_ui.set_plot_bounds(bounds);
            }
            for (name, data, colour, shape) in series {
                let color = colour(is_dark);
                plot_ui.line(Line::new(name, data).color(color).width(3.0_f32));
                // Same name, so one legend entry toggles line and markers together.
                let points = Points::new(name, data).color(color).shape(shape);
                plot_ui.points(points.radius(4.0_f32));
            }
        });
    }

    /// Axes for `shown`, cut where w(R) drops below the wall.
    fn bounds(&self, shown: Shown) -> Option<PlotBounds> {
        if !self.finished || !shown.is_cut() {
            return None;
        }
        let series: Vec<&[PlotPoint]> = self.series(shown).iter().map(|s| s.1).collect();
        fit_bounds(&self.free_energy, &series, WALL)
    }

    fn series(&self, shown: Shown) -> Vec<Series<'_>> {
        let blue: Colour = curve_color;
        let one = |name, data| vec![(name, data, blue, MarkerShape::Circle)];
        match shown {
            Shown::FreeEnergy => vec![
                ("w(R)", &self.free_energy[..], blue, MarkerShape::Circle),
                ("⟨U⟩(R)", &self.mean_energy[..], red, MarkerShape::Diamond),
                ("−TS(R)", &self.entropy[..], green, MarkerShape::Square),
            ],
            Shown::MeanForce => one("−dw/dR", &self.mean_force[..]),
            Shown::HeatCapacity => one("C(R)/k", &self.heat_capacity[..]),
            Shown::VirialRunning => one("B₂₂(R)", &self.b2[..]),
            Shown::B2Start => one("B₂₂(R′)", &self.b2_start[..]),
        }
    }
}

fn plot_points(xy: Vec<(f64, f64)>) -> Vec<PlotPoint> {
    xy.into_iter().map(|(x, y)| PlotPoint::new(x, y)).collect()
}

/// Axes from the first R where w(R) < `wall` to the last R, so the steep
/// contact wall doesn't squash the well; y spans every series in that range.
fn fit_bounds(free_energy: &[PlotPoint], series: &[&[PlotPoint]], wall: f64) -> Option<PlotBounds> {
    let from = free_energy.iter().find(|p| p.y < wall)?.x;
    let to = free_energy.last()?.x;
    let ys = series
        .iter()
        .flat_map(|s| s.iter())
        .filter(|p| (from..=to).contains(&p.x) && p.y.is_finite())
        .map(|p| p.y);
    let (low, high) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), y| {
        (l.min(y), h.max(y))
    });
    if from >= to || low >= high {
        return None;
    }
    let pad = 0.05 * (high - low);
    Some(PlotBounds::from_min_max(
        [from, low - pad],
        [to, high + pad],
    ))
}

/// −dw/dR (kT/Å) by central difference; the end points lack a neighbour.
fn mean_force(w: &[PlotPoint]) -> Vec<PlotPoint> {
    w.windows(3)
        .map(|p| PlotPoint::new(p[1].x, -(p[2].y - p[0].y) / (p[2].x - p[0].x)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_b2_appears_on_finish_and_ends_at_b2() {
        let archive = crate::test_archive::archive();
        let mut curves = Curves::new();
        archive.points.iter().for_each(|&p| curves.push(p));
        assert!(curves.b2.is_empty());

        curves.finish(&archive);
        let last = curves.b2.last().expect("a point per R ≥ σ").y;
        assert!((last - archive.virial.b2()).abs() < 1e-9 * archive.virial.b2().abs());
    }

    fn points(xy: &[(f64, f64)]) -> Vec<PlotPoint> {
        xy.iter().map(|&(x, y)| PlotPoint::new(x, y)).collect()
    }

    /// The 50 kT contact point is cut; y spans −2…5 kT padded by 5 % of 7 kT.
    #[test]
    fn axes_start_below_the_wall() {
        let w = points(&[(10.0, 50.0), (11.0, 5.0), (12.0, -2.0), (13.0, 0.0)]);
        let bounds = fit_bounds(&w, &[&w], 10.0).expect("w dips below the wall");
        assert_eq!([bounds.min()[0], bounds.max()[0]], [11.0, 13.0]);
        let [low, high] = [bounds.min()[1], bounds.max()[1]];
        assert!(
            (low + 2.35).abs() < 1e-12 && (high - 5.35).abs() < 1e-12,
            "{low} {high}"
        );
    }

    /// Nothing to cut, or nothing to show: leave the axes to auto-fit.
    #[test]
    fn no_fit_without_a_well_or_data() {
        let wall = points(&[(10.0, 50.0), (11.0, 20.0)]);
        assert!(fit_bounds(&wall, &[&wall], 10.0).is_none());
        let w = points(&[(10.0, 50.0), (11.0, 5.0)]);
        assert!(
            fit_bounds(&w, &[&[]], 10.0).is_none(),
            "B₂₂ before the scan ends"
        );
    }

    /// Mean force has no point at the first R, yet starts where w does.
    #[test]
    fn every_view_cuts_where_w_crosses_the_wall() {
        let mut curves = Curves::new();
        curves.finished = true;
        curves.free_energy = points(&[(10.0, 50.0), (11.0, 5.0), (12.0, -2.0), (13.0, 0.0)]);
        curves.mean_force = points(&[(11.0, 30.0), (12.0, 2.5)]);
        let bounds = curves.bounds(Shown::MeanForce).unwrap();
        assert_eq!([bounds.min()[0], bounds.max()[0]], [11.0, 13.0]);
        assert_eq!(bounds.max()[1], 30.0 + 0.05 * 27.5);
    }

    /// A streaming scan keeps auto-bounds so the plot grows with it; B₂₂ by
    /// start is about the small-R end, so it is never cut.
    #[test]
    fn only_finished_scans_cut_and_never_b2_by_start() {
        let archive = crate::test_archive::archive();
        let mut curves = Curves::new();
        archive.points.iter().for_each(|&p| curves.push(p));
        assert!(
            curves.bounds(Shown::FreeEnergy).is_none(),
            "still streaming"
        );
        curves.finish(&archive);
        assert!(curves.bounds(Shown::B2Start).is_none());
    }

    /// A lone point or flat line has no extent to fit; auto-bounds handles it.
    #[test]
    fn degenerate_ranges_are_not_fitted() {
        let last_only = points(&[(10.0, 50.0), (11.0, 0.0)]);
        assert!(fit_bounds(&last_only, &[&last_only], 10.0).is_none());
        let flat = points(&[(10.0, 0.0), (11.0, 0.0)]);
        assert!(fit_bounds(&flat, &[&flat], 10.0).is_none());
    }

    #[test]
    fn mean_force_of_a_linear_free_energy_is_constant() {
        let w: Vec<_> = (10..16)
            .map(|r| PlotPoint::new(r, 0.7 * r as f64))
            .collect();
        let force = mean_force(&w);
        let rs: Vec<f64> = force.iter().map(|p| p.x).collect();
        assert_eq!(rs, [11.0, 12.0, 13.0, 14.0]);
        for f in force.iter().map(|p| p.y) {
            assert!((f + 0.7).abs() < 1e-12, "{f}");
        }
    }
}
