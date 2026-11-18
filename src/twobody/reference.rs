//! B₂ʰˢ, the normaliser in B₂* = B₂/B₂ʰˢ. Display only: K_d, ΔA° and σ_eff
//! keep the scan's steric reference, so no absolute number follows this choice.

use crate::widgets::row;
use duello::VirialCoeff;

/// Which B₂ʰˢ divides B₂. Experiments choose their own, so a comparison
/// needs the same one, unless it compares absolute B₂.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Normaliser {
    /// The scan's hard-bead B₂ʳᵉᶠ, from bead positions and radii.
    #[default]
    Steric,
    /// Hard spheres of these radii (Å), touching at their sum.
    Spheres { radii: [f64; 2] },
    /// Typed in (Å³), e.g. to match a paper.
    Custom { b2_hs: f64 },
}

/// B₂ seen through one normaliser.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduced {
    pub b2_star: f64,
    /// Å³
    pub b2_hs: f64,
    /// What B₂ʰˢ is, for captions and the CSV.
    pub label: String,
}

/// As duello's fallback compares, relative.
const SAME: f64 = 1e-9;

impl Normaliser {
    /// A homodimer's spheres are one radius, A's.
    pub fn reduce(&self, virial: &VirialCoeff, homodimer: bool) -> Reduced {
        let (b2_hs, label) = match *self {
            Normaliser::Steric => (virial.reference(), steric_label(virial)),
            Normaliser::Spheres { radii: [a, b] } => {
                let b = if homodimer { a } else { b };
                let label = match homodimer {
                    true => format!("spheres, a = {a} Å"),
                    false => format!("spheres, a = {a}, {b} Å"),
                };
                (hard_sphere(a + b), label)
            }
            Normaliser::Custom { b2_hs } => (b2_hs, "custom".into()),
        };
        Reduced {
            b2_star: virial.b2() / b2_hs,
            b2_hs,
            label,
        }
    }
}

/// Shared by the label and the combo's entries.
const TIP: &str = "B₂ʰˢ, the normaliser in B₂* = B₂/B₂ʰˢ. Experiments choose their own, so \
     a comparison with experiment needs the same one, unless it compares absolute B₂. \
     Display only: K_d, ΔA°, ΔU°, ΔS° and σ_eff keep the steric reference, and changing \
     this needs no new scan.";

impl Normaliser {
    fn name(&self) -> &'static str {
        match self {
            Normaliser::Steric => "Steric",
            Normaliser::Spheres { .. } => "Spheres",
            Normaliser::Custom { .. } => "Custom",
        }
    }

    fn tip(&self) -> &'static str {
        match self {
            Normaliser::Steric => {
                "The scan's hard-bead B₂ʳᵉᶠ, from bead positions and radii; it doesn't \
                 depend on pH, salt or temperature"
            }
            Normaliser::Spheres { .. } => {
                "Hard spheres of the radii given: (2π/3)(a_A + a_B)³, or 16πa³/3 for a homodimer"
            }
            Normaliser::Custom { .. } => "A value typed in, e.g. to match a paper",
        }
    }

    /// Rows of the Scan range grid: the choice, then its numbers if any.
    pub fn ui(&mut self, ui: &mut egui::Ui, homodimer: bool) {
        let choices = [
            Normaliser::Steric,
            Normaliser::Spheres { radii: [15.0; 2] },
            Normaliser::Custom { b2_hs: 1.75e5 },
        ];
        row(ui, "B₂ʰˢ", TIP, |ui| {
            egui::ComboBox::from_id_salt("normaliser")
                .selected_text(self.name())
                .show_ui(ui, |ui| {
                    for choice in choices {
                        let picked =
                            std::mem::discriminant(self) == std::mem::discriminant(&choice);
                        if ui
                            .selectable_label(picked, choice.name())
                            .on_hover_text(choice.tip())
                            .clicked()
                            && !picked
                        {
                            *self = choice;
                        }
                    }
                })
                .response
        });
        self.ui_values(ui, homodimer);
    }

    fn ui_values(&mut self, ui: &mut egui::Ui, homodimer: bool) {
        match self {
            Normaliser::Steric => {}
            Normaliser::Spheres { radii: [a, b] } => {
                let tip = "Sphere radii of molecules A and B";
                row(ui, "", tip, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(radius(a)).on_hover_text("Radius of A");
                        if !homodimer {
                            ui.add(radius(b)).on_hover_text("Radius of B");
                        }
                    })
                    .response
                });
            }
            Normaliser::Custom { b2_hs } => {
                row(ui, "", "B₂ʰˢ in Å³", |ui| {
                    ui.add(
                        egui::DragValue::new(b2_hs)
                            .speed(100.0)
                            .range(1.0..=1e12)
                            .suffix(" Å³"),
                    )
                });
            }
        }
    }
}

/// 2πσ³/3 for contact distance σ, as duello computes it.
fn hard_sphere(sigma: f64) -> f64 {
    VirialCoeff::from_raw_parts(0.0, sigma).reference()
}

fn radius(a: &mut f64) -> egui::DragValue<'_> {
    egui::DragValue::new(a)
        .speed(0.1)
        .range(0.1..=500.0)
        .suffix(" Å")
}

/// Without bead radii, duello's reference is the hard sphere at σ, which moves
/// with R_min: say so rather than "steric".
pub fn steric_label(virial: &VirialCoeff) -> String {
    let sigma = virial.sigma();
    let hard_sphere = hard_sphere(sigma);
    match ((virial.reference() - hard_sphere) / hard_sphere).abs() <= SAME {
        true => format!("hard sphere at σ = {sigma:.1} Å (no bead radii)"),
        false => "steric".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn b2_hs(normaliser: Normaliser, homodimer: bool) -> f64 {
        let virial = crate::test_archive::archive().virial.with_reference(5000.0);
        normaliser.reduce(&virial, homodimer).b2_hs
    }

    #[test]
    fn spheres_touch_at_the_sum_of_their_radii() {
        let spheres = |a, b| Normaliser::Spheres { radii: [a, b] };
        assert!((b2_hs(spheres(10.0, 10.0), false) - 16.0 * PI * 1000.0 / 3.0).abs() < 1e-6);
        assert!((b2_hs(spheres(10.0, 20.0), false) - 2.0 * PI * 27000.0 / 3.0).abs() < 1e-6);
        // A homodimer has one radius; B's field is hidden and ignored.
        assert!((b2_hs(spheres(10.0, 99.0), true) - 16.0 * PI * 1000.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn custom_and_steric_give_their_own_value() {
        assert_eq!(b2_hs(Normaliser::Custom { b2_hs: 1.75e5 }, true), 1.75e5);
        assert_eq!(b2_hs(Normaliser::Steric, true), 5000.0);
        let virial = crate::test_archive::archive().virial.with_reference(5000.0);
        let reduced = Normaliser::Custom {
            b2_hs: 2.0 * virial.b2(),
        }
        .reduce(&virial, true);
        assert!((reduced.b2_star - 0.5).abs() < 1e-12);
    }

    /// Without bead radii duello falls back to the hard sphere at σ, which moves
    /// with R_min; the label must not call that "steric".
    #[test]
    fn the_hard_sphere_fallback_says_so() {
        let fallback = crate::test_archive::archive().virial;
        let label = Normaliser::Steric.reduce(&fallback, true).label;
        assert!(label.contains("hard sphere at σ"), "{label}");
        let steric = fallback.with_reference(5000.0);
        assert_eq!(Normaliser::Steric.reduce(&steric, true).label, "steric");
    }
}
