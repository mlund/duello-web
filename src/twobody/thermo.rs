//! The Thermodynamics window: what the angular partition sum gives beyond the PMF.

use crate::widgets::{grid, row};
use duello::archive::ScanArchive;
use duello::{open_at_sigma, weight_at_sigma, Association, VirialCoeff, OPEN_AT_SIGMA};

/// Every quantity here comes from the PMF at a fixed pair potential, so T-derivatives
/// leave out how εᵣ, κ and titration change with temperature.
const FIXED_POTENTIAL: &str = "At a fixed pair potential: leaves out how the permittivity, \
     the Debye length and titration change with temperature.";

pub fn thermodynamics(ui: &mut egui::Ui, archive: &ScanArchive) {
    grid("twobody_thermo").striped(true).show(ui, |ui| {
        virial_rows(ui, archive);
        if let Some(association) = Association::new(&archive.virial, &archive.points) {
            association_rows(ui, &association, archive.settings.temperature);
        }
    });
}

fn rows(ui: &mut egui::Ui, rows: &[(&str, String, String)]) {
    for (label, tip, text) in rows {
        row(ui, label, tip, |ui| ui.label(text));
    }
}

fn virial_rows(ui: &mut egui::Ui, archive: &ScanArchive) {
    let virial = &archive.virial;
    let (mw1, mw2) = archive.molar_masses();
    let sigma = virial.sigma();
    let slope = VirialCoeff::temperature_slope(&archive.points, sigma);
    let b2_tip = "Osmotic second virial coefficient of the protein pair, McMillan–Mayer: \
         solvent and salt are implicit, at fixed solvent and salt chemical potentials. \
         Positive: net repulsion; negative: net attraction. Hard spheres below σ.";
    rows(
        ui,
        &[
            ("B₂₂", b2_tip.into(), format!("{:.3e} Å³", virial.b2())),
            (
                "B₂₂",
                format!("{b2_tip} Scaled by the molar masses, {mw1:.0} and {mw2:.0} g/mol."),
                format!("{:.3e} mL·mol/g²", virial.mol_ml_per_gram2(mw1, mw2)),
            ),
            (
                "B₂₂ʳᵉᶠ",
                "Second virial coefficient of the same pair with only hard-bead excluded \
                 volume (bead radii from the topology), averaged over orientations; \
                 needs no σ."
                    .into(),
                format!("{:.3e} Å³", virial.reference()),
            ),
            (
                "B₂₂/B₂₂ʳᵉᶠ",
                "Reduced: B₂₂ over the hard-bead reference B₂₂ʳᵉᶠ. Below 1: attraction \
                 beyond the beads' own excluded volume."
                    .into(),
                format!("{:.2}", virial.reduced()),
            ),
            (
                "σ_eff",
                "Diameter of hard spheres with the same B₂₂ as the hard-bead pair, \
                 (3B₂₂ʳᵉᶠ/2π)^⅓."
                    .into(),
                format!("{:.1} Å", virial.sigma_eff()),
            ),
            (
                "σ",
                "Closest separation scanned, where the B₂₂ integral starts; below it the \
                 pair counts as hard spheres. Only the integration start: the reduced \
                 B₂₂ and K_d use B₂₂ʳᵉᶠ instead."
                    .into(),
                format!("{sigma:.1} Å"),
            ),
        ],
    );
    opening_row(ui, weight_at_sigma(&archive.points, sigma));
    rows(
        ui,
        &[
            (
                "G₂₂",
                "Kirkwood–Buff integral of the pair at infinite dilution, implicit solvent \
                 (McMillan–Mayer): G₂₂ = −2B₂₂. Positive: net accumulation of a second \
                 protein around the first; negative: net exclusion, excluded volume included."
                    .into(),
                format!("{:.3e} Å³", -2.0 * virial.b2()),
            ),
            (
                "T·∂B₂₂/∂T",
                format!(
                    "How B₂₂ shifts with temperature, from ∂(w/kT)/∂(1/kT) = ⟨U⟩. \
                     Positive: B₂₂ rises on heating. {FIXED_POTENTIAL}"
                ),
                slope.map_or("–".into(), |s| format!("{s:.3e} Å³")),
            ),
        ],
    );
}

/// Red where the pair still meets at σ, since B₂₂ then depends on R_min.
fn opening_row(ui: &mut egui::Ui, weight: Option<f64>) {
    let tip = format!(
        "Boltzmann weight at σ. Below σ the pair counts as hard spheres, \
         which is exact only if this is ≈ 0. Red from {OPEN_AT_SIGMA}: lower R_min and \
         rescan, or refine dR if closer R overlap. \"B₂₂ by start\" in the Free energy \
         window shows where B₂₂ levels off."
    );
    let text = weight.map_or(egui::RichText::new("–"), |w| {
        let text = egui::RichText::new(format!("{w:.3}"));
        match open_at_sigma(w) {
            true => text.color(ui.visuals().error_fg_color),
            false => text,
        }
    });
    row(ui, "e^(−w/kT) at σ", &tip, |ui| ui.label(text));
}

/// Shown only where K_a > 0, i.e. net attraction beyond the hard-bead reference.
fn association_rows(ui: &mut egui::Ui, association: &Association, temperature: f64) {
    let standard = "Standard state 1 M.";
    let molar = association.molar(temperature);
    rows(
        ui,
        &[
            (
                "K_d",
                "Dissociation constant 1/K_a, with K_a = −2(B₂₂ − B₂₂ʳᵉᶠ): the attraction \
                 beyond the beads' own excluded volume, with no contact criterion for a \
                 complex. It depends on how the reference is defined; B₂₂ does not (Hill)."
                    .into(),
                format!("{:.2e} M", association.ka.recip()),
            ),
            (
                "ΔA°",
                format!("Association free energy, −kT·ln(K_a·1 M). {standard}"),
                format!("{:.2} kJ/mol", molar.delta_a),
            ),
            (
                "ΔU°",
                format!("Association energy, −∂ln K_a/∂(1/kT). {standard} {FIXED_POTENTIAL}"),
                format!("{:.2} kJ/mol", molar.delta_u),
            ),
            (
                "ΔS°",
                format!(
                    "Association entropy, (ΔU° − ΔA°)/T at {temperature:.2} K. \
                     {standard} {FIXED_POTENTIAL}"
                ),
                format!("{:.1} J/(K·mol)", molar.delta_s),
            ),
        ],
    );
}
