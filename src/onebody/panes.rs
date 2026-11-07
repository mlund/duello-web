//! What the Multipole and Titration windows draw from a titrated molecule.

use crate::widgets::{grid, row};
use duello::molecule::{CurvePoint, Molecule};
use egui_plot::{Line, Plot, PlotPoints, VLine};

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

/// Z(pH) and capacitance(pH), marking `ph`; the curve uses Henderson–Hasselbalch
/// as MC is too slow for a whole curve.
pub fn titration_plot(ui: &mut egui::Ui, curve: &[CurvePoint], ph: f64, height: f32) {
    let series = |f: fn(&CurvePoint) -> f64| {
        PlotPoints::from(curve.iter().map(|p| [p.ph, f(p)]).collect::<Vec<_>>())
    };
    let marker = ui.visuals().weak_text_color();
    Plot::new("titration_curve")
        .height(height)
        .legend(crate::widgets::curve_legend())
        .x_axis_label("pH")
        .y_axis_label("Z (e), capacitance (e²)")
        .show(ui, |plot| {
            plot.line(Line::new("Z", series(|p| p.charge)));
            plot.line(
                Line::new("capacitance", series(|p| p.capacitance))
                    .style(egui_plot::LineStyle::dashed_dense()),
            );
            plot.vline(VLine::new("current pH", ph).color(marker));
        })
        .response
        .on_hover_text(
            "Net charge and its fluctuation from Henderson–Hasselbalch. The site table \
             and Multipole come from Monte Carlo, which adds site–site coupling.",
        );
}
