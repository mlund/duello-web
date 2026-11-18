//! The Thermodynamics window: what the angular partition sum gives beyond the PMF.

use super::summary::{Quantity, Shown, Summary};
use crate::widgets::{grid, row};

pub fn thermodynamics(ui: &mut egui::Ui, summary: &Summary, homodimer: bool) {
    grid("twobody_thermo").striped(true).show(ui, |ui| {
        for q in summary.thermo.iter().filter(|q| shown(q)) {
            row(ui, &q.label, &q.tip, |ui| ui.label(text(ui, q)));
        }
        molecule_rows(ui, summary, homodimer);
    });
}

fn shown(q: &Quantity) -> bool {
    match q.shown {
        Shown::Always => true,
        Shown::IfSet => q.value.is_some(),
        Shown::CsvOnly => false,
    }
}

fn text(ui: &egui::Ui, q: &Quantity) -> egui::RichText {
    let text = egui::RichText::new(q.text());
    match q.warn {
        true => text.color(ui.visuals().error_fg_color),
        false => text,
    }
}

/// One row per moment, A and B side by side; a homodimer's are one molecule's.
fn molecule_rows(ui: &mut egui::Ui, summary: &Summary, homodimer: bool) {
    ui.strong("Molecules")
        .on_hover_text("Each molecule as scanned, at the scan's pH, salt and temperature");
    ui.label(if homodimer { "" } else { "A / B" })
        .on_hover_text("Molecule 1 / molecule 2");
    ui.end_row();
    let [a, b] = &summary.molecules;
    for (qa, qb) in a.iter().zip(b) {
        let text = match homodimer {
            true => qa.text(),
            false => format!("{} / {}", qa.text(), qb.text()),
        };
        row(ui, &qa.label, &qa.tip, |ui| ui.label(text));
    }
}

#[cfg(test)]
mod tests {
    use crate::twobody::reference::Normaliser;
    use crate::twobody::summary::Facts;

    #[test]
    fn the_pane_renders_with_and_without_moments() {
        let mut archive = crate::test_archive::associating();
        archive.bodies[1].multipole = Some(crate::test_archive::multipole());
        let summary = Facts::of(&archive).summary(&Normaliser::Steric);
        for homodimer in [true, false] {
            crate::widgets::headless(|ui| super::thermodynamics(ui, &summary, homodimer));
        }
    }
}
