//! What the Multipole and Titration windows draw.

use crate::widgets::{grid, row};
use duello::molecule::{CurvePoint, Molecule};
use egui_plot::{Line, Plot, PlotPoints, Points, VLine};

/// 1 e·Å in debye.
const DEBYE_PER_E_ANGSTROM: f64 = 4.803_204;

pub fn multipole(ui: &mut egui::Ui, molecule: &Molecule) {
    let m = &molecule.multipole;
    let mass = molecule.body.mass();
    let sites = molecule
        .beads
        .iter()
        .filter(|b| b.charge_variance.is_some())
        .count();
    let rows = [
        (
            "Net charge Z",
            "Mean net charge ⟨Z⟩",
            format!("{:+.2} e", m.charge),
        ),
        (
            "Capacitance",
            "Charge fluctuation ⟨Z²⟩ − ⟨Z⟩², e²: how readily the charge responds to a \
             nearby charge",
            format!("{:.2} e²", m.capacitance()),
        ),
        (
            "Dipole μ",
            "Mean dipole moment about the mass centre",
            format!(
                "{:.1} e·Å = {:.0} D",
                m.dipole,
                m.dipole * DEBYE_PER_E_ANGSTROM
            ),
        ),
        (
            "σ_μ",
            "Dipole fluctuation from titration, √(⟨μ²⟩ − ⟨μ⟩²)",
            format!("{:.1} e·Å", m.dipole_std()),
        ),
        (
            "Quadrupole",
            "Mean quadrupole magnitude about the mass centre",
            format!("{:.3e} e·Å²", m.quadrupole),
        ),
        (
            "σ_Q",
            "Quadrupole fluctuation from titration",
            format!("{:.3e} e·Å²", m.quadrupole_std()),
        ),
        ("Mass", "Sum of bead masses", format!("{mass:.0} g/mol")),
        (
            "Beads",
            "Coarse-grained beads",
            molecule.body.positions.len().to_string(),
        ),
        (
            "Titratable sites",
            "Beads whose charge titrates",
            sites.to_string(),
        ),
    ];
    grid("multipole").striped(true).show(ui, |ui| {
        for (label, tip, value) in rows {
            row(ui, label, tip, |ui| ui.label(value));
        }
    });
}

/// What the titration plot shows over pH.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    Charge,
    Dipole,
    Quadrupole,
    Capacitance,
}

impl Quantity {
    const ALL: [Quantity; 4] = [
        Quantity::Charge,
        Quantity::Dipole,
        Quantity::Quadrupole,
        Quantity::Capacitance,
    ];

    fn label(self) -> &'static str {
        match self {
            Quantity::Charge => "Charge",
            Quantity::Dipole => "Dipole",
            Quantity::Quadrupole => "Quadrupole",
            Quantity::Capacitance => "Capacitance",
        }
    }

    fn axis(self) -> &'static str {
        match self {
            Quantity::Charge => "⟨Z⟩ (e)",
            Quantity::Dipole => "⟨μ⟩ (e·Å)",
            Quantity::Quadrupole => "⟨Q⟩ (e·Å²)",
            Quantity::Capacitance => "⟨Z²⟩ − ⟨Z⟩² (e²)",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Quantity::Charge => "Mean net charge ⟨Z⟩",
            Quantity::Dipole => "Mean dipole moment ⟨μ⟩ about the mass centre",
            Quantity::Quadrupole => {
                "Mean quadrupole ⟨Q⟩ = ⟨‖Θ‖⟩, the norm of the traceless (Buckingham) \
                 tensor about the mass centre"
            }
            Quantity::Capacitance => {
                "Charge fluctuation: how readily the charge responds to a nearby charge"
            }
        }
    }

    fn of(self, p: &CurvePoint) -> f64 {
        match self {
            Quantity::Charge => p.charge,
            Quantity::Dipole => p.dipole,
            Quantity::Quadrupole => p.quadrupole,
            Quantity::Capacitance => p.capacitance,
        }
    }
}

/// `quantity` over pH from Monte Carlo (`mc`, filling in) and ideal
/// Henderson–Hasselbalch (`ideal`), marking the current `ph`.
pub fn titration(
    ui: &mut egui::Ui,
    quantity: &mut Quantity,
    reset: &mut bool,
    ideal: &[CurvePoint],
    mc: &[CurvePoint],
    ph: f64,
) {
    let before = *quantity;
    crate::widgets::pick(
        ui,
        "titration_quantity",
        quantity,
        &Quantity::ALL,
        |q| q.label(),
        Some(Quantity::tooltip),
    )
    .on_hover_text("What to plot over pH");
    // Quantities differ by orders of magnitude, so old axes would hide the new curve.
    *reset |= *quantity != before;
    let q = *quantity;
    let series =
        |curve: &[CurvePoint]| -> Vec<[f64; 2]> { curve.iter().map(|p| [p.ph, q.of(p)]).collect() };
    let (mc, ideal) = (series(mc), series(ideal));
    let marker = ui.visuals().weak_text_color();
    let plot = Plot::new("titration_curve")
        .legend(crate::widgets::curve_legend())
        .x_axis_label("pH")
        .y_axis_label(q.axis());
    crate::widgets::refit(plot, reset)
        .show(ui, |plot| {
            plot.line(Line::new("Monte Carlo", PlotPoints::from(mc.clone())));
            plot.points(Points::new("Monte Carlo", PlotPoints::from(mc)).radius(2.5));
            plot.line(
                Line::new("ideal (HH)", PlotPoints::from(ideal))
                    .style(egui_plot::LineStyle::dashed_dense()),
            );
            plot.vline(VLine::new("current pH", ph).color(marker));
        })
        .response
        .on_hover_text(
            "Monte Carlo includes site–site coupling; ideal Henderson–Hasselbalch \
             titrates each site alone. Monte Carlo points fill in as they finish.",
        );
}
