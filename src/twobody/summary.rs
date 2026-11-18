//! One scan's numbers, named and unit-tagged once, so the Thermodynamics pane,
//! the Solution scan plot and its CSV can't disagree.

use super::reference::{Normaliser, Reduced};
use duello::archive::{CgBody, Multipole, ScanArchive};
use duello::{weight_at_sigma, Association, VirialCoeff};

/// What the pane and a sweep's row need from a scan, without its table, so a
/// sweep can keep many points.
#[derive(Clone, Debug)]
pub struct Facts {
    pub ph: f64,
    /// mol/L
    pub ionic_strength: f64,
    /// K
    pub temperature: f64,
    pub homodimer: bool,
    pub virial: VirialCoeff,
    /// T·∂B₂/∂T, Å³
    pub slope: Option<f64>,
    /// None where K_a ≤ 0.
    pub association: Option<Association>,
    pub weight_at_sigma: Option<f64>,
    pub bodies: [BodySummary; 2],
}

/// One molecule as scanned. Moments are None for fixed-charge (XYZ) input,
/// which had no titration to sample them.
#[derive(Clone, Debug, PartialEq)]
pub struct BodySummary {
    /// ⟨Z⟩, e
    pub charge: f64,
    /// ⟨Z²⟩ − ⟨Z⟩², e²
    pub capacitance: Option<f64>,
    /// ⟨|μ|⟩, e·Å
    pub dipole: Option<f64>,
    /// ⟨‖Θ‖⟩, e·Å²
    pub quadrupole: Option<f64>,
    /// g/mol
    pub molar_mass: f64,
}

impl Facts {
    pub fn of(archive: &ScanArchive) -> Self {
        let virial = archive.virial.clone();
        let sigma = virial.sigma();
        let settings = &archive.settings;
        Self {
            ph: settings.ph,
            ionic_strength: settings.ionic_strength,
            temperature: settings.temperature,
            homodimer: settings.homo_dimer,
            slope: VirialCoeff::temperature_slope(&archive.points, sigma),
            association: Association::new(&virial, &archive.points),
            weight_at_sigma: weight_at_sigma(&archive.points, sigma),
            bodies: archive.bodies.each_ref().map(BodySummary::of),
            virial,
        }
    }

    /// The quantities with B₂* under `normaliser`.
    pub fn summary(&self, normaliser: &Normaliser) -> Summary {
        Summary::of(self, &normaliser.reduce(&self.virial, self.homodimer))
    }
}

impl BodySummary {
    fn of(body: &CgBody) -> Self {
        let m = body.multipole.as_ref();
        Self {
            charge: m.map_or_else(|| body.net_charge(), |m| m.charge),
            capacitance: m.map(Multipole::capacitance),
            dipole: m.map(|m| m.dipole),
            quadrupole: m.map(|m| m.quadrupole),
            molar_mass: body.mass(),
        }
    }
}

/// Every quantity here comes from the PMF at a fixed pair potential, so T-derivatives
/// leave out how εᵣ, κ and titration change with temperature.
pub const FIXED_POTENTIAL: &str = "At a fixed pair potential: leaves out how the \
     permittivity, the Debye length and titration change with temperature.";

/// How the pane writes a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    /// Significant digits after the point, in scientific notation.
    Sci(usize),
    Fixed(usize),
}

/// Which quantity; its CSV name is [`Key::name`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    T,
    Ph,
    I,
    B2,
    B2Mass,
    B2Star,
    B2Hs,
    B2Ref,
    SigmaEff,
    Sigma,
    WSigma,
    G2,
    TdB2dT,
    Kd,
    DA,
    DU,
    DS,
    Z,
    C,
    Mu,
    Theta,
    Mw,
}

impl Key {
    /// CSV column name, ASCII where possible.
    pub fn name(self) -> &'static str {
        match self {
            Key::T => "T",
            Key::Ph => "pH",
            Key::I => "I",
            Key::B2 => "B2",
            Key::B2Mass => "B2_mass",
            Key::B2Star => "B2*",
            Key::B2Hs => "B2hs",
            Key::B2Ref => "B2ref",
            Key::SigmaEff => "sigma_eff",
            Key::Sigma => "sigma",
            Key::WSigma => "w_sigma",
            Key::G2 => "G2",
            Key::TdB2dT => "TdB2dT",
            Key::Kd => "Kd",
            Key::DA => "dA",
            Key::DU => "dU",
            Key::DS => "dS",
            Key::Z => "Z",
            Key::C => "C",
            Key::Mu => "mu",
            Key::Theta => "Theta",
            Key::Mw => "Mw",
        }
    }
}

/// Where a quantity appears.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shown {
    /// Pane and CSV.
    Always,
    /// The pane hides it while empty, e.g. K_d without attraction.
    IfSet,
    /// Shown elsewhere in the app, e.g. the conditions in Onebody.
    CsvOnly,
}

/// One named value with its unit; empty where it doesn't apply, e.g. K_d
/// without association.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantity {
    pub key: Key,
    /// Empty for a pure number.
    pub unit: &'static str,
    /// Pane label, e.g. "B₂₂".
    pub label: String,
    pub tip: String,
    pub value: Option<f64>,
    pub format: Format,
    pub shown: Shown,
    /// Drawn in the error colour: the value undermines another, e.g. B₂ at σ.
    pub warn: bool,
}

impl Quantity {
    fn new(key: Key, unit: &'static str, label: impl Into<String>, value: Option<f64>) -> Self {
        Self {
            key,
            unit,
            label: label.into(),
            tip: String::new(),
            value,
            format: Format::Sci(3),
            shown: Shown::Always,
            warn: false,
        }
    }

    fn tip(self, tip: impl Into<String>) -> Self {
        Self {
            tip: tip.into(),
            ..self
        }
    }

    fn format(self, format: Format) -> Self {
        Self { format, ..self }
    }

    fn shown(self, shown: Shown) -> Self {
        Self { shown, ..self }
    }

    fn warn(self, warn: bool) -> Self {
        Self { warn, ..self }
    }

    /// As the pane shows it, with its unit; "–" when empty.
    pub fn text(&self) -> String {
        let Some(v) = self.value else {
            return "–".into();
        };
        let number = match self.format {
            Format::Sci(d) => format!("{v:.d$e}"),
            Format::Fixed(d) => format!("{v:.d$}"),
        };
        match self.unit {
            "" => number,
            unit => format!("{number} {unit}"),
        }
    }

    /// CSV column name, e.g. "Kd/M", or "ZA/e" for molecule "A"; a unit with
    /// its own slash goes in parentheses, e.g. "dA/(kJ/mol)".
    pub fn header(&self, molecule: &str) -> String {
        match self.unit {
            "" => format!("{}{molecule}", self.key.name()),
            unit if unit.contains('/') => format!("{}{molecule}/({unit})", self.key.name()),
            unit => format!("{}{molecule}/{unit}", self.key.name()),
        }
    }

    /// Shortest text that reads back to the same number, in e-notation outside
    /// 1e-3..1e7 where plain digits run long; empty when None.
    pub fn cell(&self) -> String {
        match self.value {
            None => String::new(),
            Some(v) if v == 0.0 || (1e-3..1e7).contains(&v.abs()) => v.to_string(),
            Some(v) => format!("{v:e}"),
        }
    }
}

/// A scan's quantities: the pair's, then each molecule's.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub thermo: Vec<Quantity>,
    pub molecules: [Vec<Quantity>; 2],
}

impl Summary {
    pub fn of(facts: &Facts, reduced: &Reduced) -> Self {
        Self {
            thermo: [
                conditions(facts),
                virial(facts, reduced),
                association(facts),
            ]
            .concat(),
            molecules: facts.bodies.each_ref().map(molecule),
        }
    }

    /// A pair quantity by key.
    pub fn value(&self, key: Key) -> Option<f64> {
        self.thermo.iter().find(|q| q.key == key)?.value
    }
}

/// B₂₂ for a homodimer; B₁₂ is the cross term of two different molecules.
pub fn b2_symbol(homodimer: bool) -> &'static str {
    if homodimer {
        "B₂₂"
    } else {
        "B₁₂"
    }
}

/// Onebody shows them, so only the CSV needs them.
fn conditions(f: &Facts) -> Vec<Quantity> {
    let quantities = vec![
        Quantity::new(Key::T, "K", "T", Some(f.temperature)).format(Format::Fixed(2)),
        Quantity::new(Key::Ph, "", "pH", Some(f.ph)).format(Format::Fixed(2)),
        Quantity::new(Key::I, "M", "I", Some(f.ionic_strength)).format(Format::Fixed(4)),
    ];
    quantities
        .into_iter()
        .map(|q| q.shown(Shown::CsvOnly))
        .collect()
}

fn virial(f: &Facts, reduced: &Reduced) -> Vec<Quantity> {
    let v = &f.virial;
    let b2 = b2_symbol(f.homodimer);
    let [mw1, mw2] = f.bodies.each_ref().map(|b| b.molar_mass);
    let b2_tip = "Osmotic second virial coefficient of the protein pair, McMillan–Mayer: \
         solvent and salt are implicit, at fixed solvent and salt chemical potentials. \
         Positive: net repulsion; negative: net attraction. Hard spheres below σ.";
    let sigma = v.sigma();
    vec![
        Quantity::new(Key::B2, "Å³", b2, Some(v.b2())).tip(b2_tip),
        Quantity::new(
            Key::B2Mass,
            "mL·mol/g²",
            b2,
            Some(v.mol_ml_per_gram2(mw1, mw2)),
        )
        .tip(format!(
            "{b2_tip} Scaled by the molar masses, {mw1:.0} and {mw2:.0} g/mol."
        )),
        Quantity::new(
            Key::B2Star,
            "",
            format!("{b2}* ({})", reduced.label),
            Some(reduced.b2_star),
        )
        .format(Format::Fixed(2))
        .tip(format!(
            "Reduced: {b2} over B₂ʰˢ = {:.3e} Å³ ({}), set under Scan range. Below 1: \
                 attraction beyond that excluded volume. Display only: K_d and σ_eff keep \
                 the steric reference.",
            reduced.b2_hs, reduced.label
        )),
        Quantity::new(Key::B2Hs, "Å³", "B₂ʰˢ", Some(reduced.b2_hs))
            .shown(Shown::CsvOnly)
            .tip(format!("The normaliser of {b2}*: {}.", reduced.label)),
        Quantity::new(Key::B2Ref, "Å³", format!("{b2}ʳᵉᶠ"), Some(v.reference()))
            .tip(reference_tip(v)),
        Quantity::new(Key::SigmaEff, "Å", "σ_eff", Some(v.sigma_eff()))
            .format(Format::Fixed(1))
            .tip(format!(
                "Diameter of hard spheres with the same {b2} as the hard-bead pair, \
                 (3{b2}ʳᵉᶠ/2π)^⅓."
            )),
        Quantity::new(Key::Sigma, "Å", "σ", Some(sigma))
            .format(Format::Fixed(1))
            .tip(format!(
                "Closest separation scanned, where the {b2} integral starts; below it the \
             pair counts as hard spheres. Only the integration start: K_d uses \
             {b2}ʳᵉᶠ instead."
            )),
        Quantity::new(Key::WSigma, "", "e^(−w/kT) at σ", f.weight_at_sigma)
            .warn(f.weight_at_sigma.is_some_and(duello::open_at_sigma))
            .format(Format::Fixed(3))
            .tip(format!(
                "Boltzmann weight at σ. Below σ the pair counts as hard spheres, which is \
                 exact only if this is ≈ 0. Red from {}: lower R_min and rescan, or refine \
                 dR if closer R overlap. \"B₂₂ by start\" in the Free energy window shows \
                 where {b2} levels off.",
                duello::OPEN_AT_SIGMA
            )),
        Quantity::new(
            Key::G2,
            "Å³",
            format!("G{}", &b2["B".len()..]),
            Some(-2.0 * v.b2()),
        )
        .tip(format!(
            "Kirkwood–Buff integral of the pair at infinite dilution, implicit solvent \
             (McMillan–Mayer): G = −2{b2}. Positive: net accumulation of a second \
             protein around the first; negative: net exclusion, excluded volume included."
        )),
        Quantity::new(Key::TdB2dT, "Å³", format!("T·∂{b2}/∂T"), f.slope).tip(format!(
            "How {b2} shifts with temperature, from ∂(w/kT)/∂(1/kT) = ⟨U⟩. \
             Positive: {b2} rises on heating. {FIXED_POTENTIAL}"
        )),
    ]
}

/// Says so when duello fell back to the hard sphere at σ for want of bead radii.
fn reference_tip(v: &VirialCoeff) -> String {
    let what = super::reference::steric_label(v);
    match what.as_str() {
        "steric" => "Second virial coefficient of the same pair with only hard-bead excluded \
                     volume (bead radii from the topology), averaged over orientations; \
                     needs no σ."
            .into(),
        _ => format!("No bead radii, so this is the {what}; it moves with R_min."),
    }
}

/// Empty cells where K_a ≤ 0, so a sweep's columns stay aligned.
fn association(f: &Facts) -> Vec<Quantity> {
    let b2 = b2_symbol(f.homodimer);
    let molar = f.association.map(|a| a.molar(f.temperature));
    let standard = "Standard state 1 M.";
    let kj = Format::Fixed(2);
    vec![
        Quantity::new(Key::Kd, "M", "K_d", f.association.map(|a| a.ka.recip()))
            .format(Format::Sci(2))
            .tip(format!(
                "Dissociation constant 1/K_a, with K_a = −2({b2} − {b2}ʳᵉᶠ): the attraction \
                 beyond the beads' own excluded volume, with no contact criterion for a \
                 complex. It depends on how the reference is defined; {b2} does not (Hill). \
                 Empty without net attraction."
            )),
        Quantity::new(Key::DA, "kJ/mol", "ΔA°", molar.map(|m| m.delta_a))
            .format(kj)
            .tip(format!(
                "Association free energy, −kT·ln(K_a·1 M). {standard}"
            )),
        Quantity::new(Key::DU, "kJ/mol", "ΔU°", molar.map(|m| m.delta_u))
            .format(kj)
            .tip(format!(
                "Association energy, −∂ln K_a/∂(1/kT). {standard} {FIXED_POTENTIAL}"
            )),
        Quantity::new(Key::DS, "J/(K·mol)", "ΔS°", molar.map(|m| m.delta_s))
            .format(Format::Fixed(1))
            .tip(format!(
                "Association entropy, (ΔU° − ΔA°)/T at {:.2} K. {standard} {FIXED_POTENTIAL}",
                f.temperature
            )),
    ]
    .into_iter()
    .map(|q| q.shown(Shown::IfSet))
    .collect()
}

fn molecule(b: &BodySummary) -> Vec<Quantity> {
    let titrated = "From the scan's own titration; empty for fixed-charge (XYZ) input.";
    vec![
        Quantity::new(Key::Z, "e", "⟨Z⟩", Some(b.charge))
            .format(Format::Fixed(2))
            .tip("Mean net charge at the scan's pH, salt and temperature"),
        Quantity::new(Key::C, "e²", "C", b.capacitance)
            .format(Format::Fixed(2))
            .tip(format!(
                "Charge capacitance ⟨Z²⟩ − ⟨Z⟩²: how readily Z responds. {titrated}"
            )),
        Quantity::new(Key::Mu, "eÅ", "⟨|μ|⟩", b.dipole)
            .format(Format::Fixed(1))
            .tip(format!(
                "Mean dipole moment magnitude about the mass centre. {titrated}"
            )),
        Quantity::new(Key::Theta, "eÅ²", "⟨‖Θ‖⟩", b.quadrupole)
            .format(Format::Fixed(0))
            .tip(format!(
                "Mean traceless quadrupole norm (Frobenius). {titrated}"
            )),
        Quantity::new(Key::Mw, "g/mol", "M_w", Some(b.molar_mass))
            .format(Format::Fixed(0))
            .tip("Molar mass of the coarse-grained beads"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_archive::associating;

    fn summary(archive: &ScanArchive) -> Summary {
        let facts = Facts::of(archive);
        let reduced = Normaliser::Steric.reduce(&facts.virial, facts.homodimer);
        Summary::of(&facts, &reduced)
    }

    #[test]
    fn kd_is_the_inverse_association_constant() {
        let archive = associating();
        let kd = summary(&archive).value(Key::Kd).unwrap();
        assert_eq!(kd, archive.virial.dissociation_const().unwrap());
    }

    /// ΔS° = (ΔU° − ΔA°)/T, in J/(K·mol) from kJ/mol.
    #[test]
    fn the_entropy_closes_the_free_energy_split() {
        let archive = associating();
        let s = summary(&archive);
        let (a, u, ds) = (s.value(Key::DA), s.value(Key::DU), s.value(Key::DS));
        let t = archive.settings.temperature;
        assert!((ds.unwrap() - 1e3 * (u.unwrap() - a.unwrap()) / t).abs() < 1e-9);
    }

    /// No association, so the K_d split is left empty rather than dropped.
    #[test]
    fn association_cells_are_empty_without_attraction() {
        let mut archive = crate::test_archive::archive();
        archive.virial = archive
            .virial
            .clone()
            .with_reference(archive.virial.b2() - 500.0);
        let s = summary(&archive);
        assert_eq!((s.value(Key::Kd), s.value(Key::DS)), (None, None));
        assert!(s.thermo.iter().any(|q| q.key == Key::Kd));
    }

    #[test]
    fn moments_come_from_the_titration_and_xyz_gives_z_only() {
        let mut archive = crate::test_archive::archive();
        archive.bodies[1].multipole = Some(crate::test_archive::multipole());
        let s = summary(&archive);
        let [a, b] = &s.molecules;
        let value = |q: &[Quantity], key| q.iter().find(|q| q.key == key).unwrap().value;
        assert_eq!(value(b, Key::Z), Some(7.5));
        assert_eq!(value(b, Key::C), Some(1.75));
        assert_eq!(value(b, Key::Mu), Some(40.0));
        assert_eq!(value(b, Key::Theta), Some(900.0));
        // Fixture beads +1 and −1, of 100 g/mol each.
        assert_eq!((value(a, Key::Z), value(a, Key::C)), (Some(0.0), None));
        assert_eq!(value(a, Key::Mw), Some(200.0));
    }

    /// Readable and still round-tripping: tiny or huge values in e-notation,
    /// and a unit with a slash in parentheses so the header splits once.
    #[test]
    fn cells_and_headers_read_unambiguously() {
        let q = |unit, value| Quantity::new(Key::Z, unit, "Z", Some(value));
        assert_eq!(q("", 3.25e-88).cell(), "3.25e-88");
        assert_eq!(q("", 4.0).cell(), "4");
        assert_eq!(q("", 734140.25).cell(), "734140.25");
        assert_eq!(q("", 2.5e7).cell(), "2.5e7");
        assert_eq!(q("kJ/mol", 1.0).header(""), "Z/(kJ/mol)");
        assert_eq!(q("e", 1.0).header("A"), "ZA/e");
    }
}
