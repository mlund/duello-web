//! The Free energy window: curves against R, streamed while a scan runs.

use crate::widgets::{curve_color, curve_legend, pick};
use duello::archive::ScanArchive;
use duello::web_api::PmfPoint;
use duello::VirialCoeff;
use egui_plot::{Line, MarkerShape, Plot, PlotPoint, Points};

/// Which curve the window plots.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Shown {
    #[default]
    FreeEnergy,
    MeanForce,
    HeatCapacity,
    VirialRunning,
}

impl Shown {
    const ALL: [Shown; 4] = [
        Shown::FreeEnergy,
        Shown::MeanForce,
        Shown::HeatCapacity,
        Shown::VirialRunning,
    ];

    fn label(&self) -> &'static str {
        match self {
            Shown::FreeEnergy => "Free energy",
            Shown::MeanForce => "Mean force",
            Shown::HeatCapacity => "Heat capacity",
            Shown::VirialRunning => "B₂₂ up to R",
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
        }
    }

    fn y_label(self) -> &'static str {
        match self {
            Shown::FreeEnergy => "Energy (kT)",
            Shown::MeanForce => "Mean force (kT/Å)",
            Shown::HeatCapacity => "C(R)/k",
            Shown::VirialRunning => "B₂₂ (Å³)",
        }
    }
}

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
        let running = VirialCoeff::running(&archive.points, archive.virial.sigma());
        self.b2 = running
            .into_iter()
            .map(|(r, b2)| PlotPoint::new(r, b2))
            .collect();
    }

    /// A picker over the curves, then the picked one's plot.
    pub fn ui(&mut self, ui: &mut egui::Ui, shown: &mut Shown, is_dark: bool) {
        let before = *shown;
        ui.horizontal(|ui| {
            ui.label("Show").on_hover_text(shown.tip());
            pick(
                ui,
                "twobody_curve",
                shown,
                &Shown::ALL,
                |s| s.label(),
                Some(Shown::tip),
            )
            .on_hover_text(shown.tip());
        });
        // Other units, so the old axes would hide the new curve.
        self.reset |= *shown != before;
        let mut plot = Plot::new("pmf_plot")
            .legend(curve_legend())
            .auto_bounds(egui::Vec2b::TRUE)
            .x_axis_label("Mass-centre separation, R (Å)")
            .y_axis_label(shown.y_label());
        // Only on these events; auto_bounds follows a growing curve in between,
        // so the user can still zoom while a scan runs.
        if std::mem::take(&mut self.reset) {
            plot = plot.reset();
        }
        let series = self.series(*shown);
        plot.show(ui, |plot_ui| {
            for (name, data, colour, shape) in series {
                let color = colour(is_dark);
                plot_ui.line(Line::new(name, data).color(color).width(3.0_f32));
                // Same name, so one legend entry toggles line and markers together.
                let points = Points::new(name, data).color(color).shape(shape);
                plot_ui.points(points.radius(4.0_f32));
            }
        });
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
        }
    }
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
