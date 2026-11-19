//! A Solution scan: one full two-body scan per pH, salt or temperature point.

use super::reference::{Normaliser, Reduced};
use super::summary::{Facts, Key, Summary};
use super::vant_hoff::VantHoff;
use duello::archive::ScanArchive;
use duello::web_api::ScanSettings;

/// The solution condition a sweep varies; the others stay at Onebody's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variable {
    Ph,
    /// Ionic strength, entered in mM.
    Salt,
    Temperature,
}

impl Variable {
    pub const ALL: [Variable; 3] = [Variable::Ph, Variable::Salt, Variable::Temperature];

    pub fn label(self) -> &'static str {
        match self {
            Variable::Ph => "pH",
            Variable::Salt => "Salt",
            Variable::Temperature => "Temperature",
        }
    }

    pub fn axis(self) -> &'static str {
        match self {
            Variable::Ph => "pH",
            Variable::Salt => "Ionic strength (mM)",
            Variable::Temperature => "Temperature (K)",
        }
    }

    /// Empty for pH.
    pub fn unit(self) -> &'static str {
        match self {
            Variable::Ph => "",
            Variable::Salt => " mM",
            Variable::Temperature => " K",
        }
    }

    pub fn tip(self) -> &'static str {
        match self {
            Variable::Ph => "Vary the pH; each point re-titrates both molecules",
            Variable::Salt => {
                "Vary the ionic strength (mM); it screens charges and shifts their titration"
            }
            Variable::Temperature => {
                "Vary the temperature (K): the Bjerrum and Debye lengths, kT and titration \
                 all follow"
            }
        }
    }

    /// As Onebody allows, in the panel's units.
    pub fn bounds(self) -> std::ops::RangeInclusive<f64> {
        match self {
            Variable::Ph => 0.0..=14.0,
            Variable::Salt => 0.0..=10_000.0,
            Variable::Temperature => 1.0..=1000.0,
        }
    }

    /// From and to when picked.
    pub fn range(self) -> (f64, f64) {
        match self {
            Variable::Ph => (4.0, 9.0),
            Variable::Salt => (5.0, 300.0),
            Variable::Temperature => (280.0, 330.0),
        }
    }

    /// `x` in the units the panel shows.
    fn set(self, settings: &mut ScanSettings, x: f64) {
        match self {
            Variable::Ph => settings.ph = x,
            Variable::Salt => settings.ionic_strength = x * 1e-3,
            Variable::Temperature => settings.temperature = x,
        }
    }
}

/// `n` points spaced linearly from `from` to `to`, both included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sweep {
    pub variable: Variable,
    pub from: f64,
    pub to: f64,
    pub n: usize,
}

/// The fewest and most points a sweep takes.
pub const POINTS: std::ops::RangeInclusive<usize> = 2..=50;

impl Default for Sweep {
    fn default() -> Self {
        Self::over(Variable::Ph)
    }
}

impl Sweep {
    /// `variable`'s default range at six points.
    pub fn over(variable: Variable) -> Self {
        let (from, to) = variable.range();
        Self {
            variable,
            from,
            to,
            n: 6,
        }
    }

    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        let step = (self.to - self.from) / (self.n.max(2) - 1) as f64;
        (0..self.n).map(move |i| self.from + step * i as f64)
    }
}

/// A sweep's finished points, and what to scan next.
pub struct SweepPoints {
    sweep: Sweep,
    base: ScanSettings,
    points: Vec<SweepPoint>,
    /// From the first point, then fixed so every point splits off the same hard core.
    rmin: Option<f64>,
}

pub struct SweepPoint {
    pub x: f64,
    pub facts: Facts,
}

/// Relative; bead radii are fixed, so only rounding may move the reference.
const SAME_REFERENCE: f64 = 1e-9;

impl SweepPoints {
    pub fn new(sweep: Sweep, base: ScanSettings) -> Self {
        Self {
            sweep,
            base,
            points: Vec::new(),
            rmin: None,
        }
    }

    /// The next point's value and settings; None once all are done.
    pub fn next(&self) -> Option<(f64, ScanSettings)> {
        let x = self.sweep.values().nth(self.points.len())?;
        let mut settings = self.base.clone();
        self.sweep.variable.set(&mut settings, x);
        if let Some(rmin) = self.rmin {
            (settings.rmin, settings.rmin_auto) = (rmin, false);
        }
        Some((x, settings))
    }

    /// Records the scan of [`Self::next`].
    pub fn push(&mut self, archive: &ScanArchive) -> anyhow::Result<()> {
        let x = (self.sweep.values().nth(self.points.len()))
            .ok_or_else(|| anyhow::anyhow!("the sweep is done"))?;
        self.check_reference(archive.virial.reference())?;
        self.rmin.get_or_insert(archive.settings.rmin);
        let facts = Facts::of(archive);
        self.points.push(SweepPoint { x, facts });
        Ok(())
    }

    pub fn points(&self) -> &[SweepPoint] {
        &self.points
    }

    pub fn sweep(&self) -> &Sweep {
        &self.sweep
    }

    /// Each point's x and quantities under `normaliser`.
    pub fn summaries(&self, normaliser: &Normaliser) -> Vec<(f64, Summary)> {
        let summary = |p: &SweepPoint| (p.x, p.facts.summary(normaliser));
        self.points.iter().map(summary).collect()
    }

    /// Comment line naming what the numbers depend on, a header with units,
    /// then one row per point; `pair` names molecules A and B.
    /// None before the first point.
    pub fn csv(&self, normaliser: &Normaliser, pair: [&str; 2]) -> Option<String> {
        let summaries = self.summaries(normaliser);
        let mut csv = self.preamble(&self.reduced(normaliser)?, pair);
        csv.push_str(&line(&summaries.first()?.1, true));
        for (_, summary) in &summaries {
            csv.push_str(&line(summary, false));
        }
        if let Some(fit) = vant_hoff(self.sweep.variable, &curve(&summaries, Key::Kd)) {
            csv.push_str(&fit.csv_comment());
        }
        Some(csv)
    }

    /// B₂ʰˢ and its wording, the same at every point (see `check_reference`).
    pub fn reduced(&self, normaliser: &Normaliser) -> Option<Reduced> {
        let first = &self.points.first()?.facts;
        Some(normaliser.reduce(&first.virial, first.homodimer))
    }

    fn preamble(&self, reduced: &Reduced, [a, b]: [&str; 2]) -> String {
        let rmin = self.rmin.unwrap_or(f64::NAN);
        format!(
            "# B₂ʰˢ = {:.3e} Å³ ({}), R_min = {rmin:.1} Å, model {}, pair {a}/{b}\n",
            reduced.b2_hs, reduced.label, self.base.model
        )
    }

    /// Every point must share the first's B₂ʳᵉᶠ, or K_d and the steric B₂*
    /// would move with the reference rather than the solution.
    fn check_reference(&self, reference: f64) -> anyhow::Result<()> {
        let Some(first) = self.points.first() else {
            return Ok(());
        };
        let expected = first.facts.virial.reference();
        anyhow::ensure!(
            (reference - expected).abs() <= SAME_REFERENCE * expected.abs(),
            "the steric reference B₂ʳᵉᶠ changed from {expected:.4e} to {reference:.4e} Å³"
        );
        Ok(())
    }
}

/// The van 't Hoff fit of `kd` as `[x, K_d]`, for temperature sweeps only.
pub fn vant_hoff(variable: Variable, kd: &[[f64; 2]]) -> Option<VantHoff> {
    (variable == Variable::Temperature)
        .then(|| VantHoff::fit(kd))
        .flatten()
}

/// `[x, value]` of `key` where it has one.
pub fn curve(summaries: &[(f64, Summary)], key: Key) -> Vec<[f64; 2]> {
    let point = |(x, s): &(f64, Summary)| Some([*x, s.value(key)?]);
    summaries.iter().filter_map(point).collect()
}

/// One CSV line: the pair's cells, then A's and B's.
fn line(summary: &Summary, header: bool) -> String {
    let [a, b] = &summary.molecules;
    let cells = (summary.thermo.iter().map(|q| (q, "")))
        .chain(a.iter().map(|q| (q, "A")))
        .chain(b.iter().map(|q| (q, "B")))
        .map(|(q, molecule)| match header {
            true => q.header(molecule),
            false => q.cell(),
        });
    cells.collect::<Vec<_>>().join(",") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ScanSettings {
        ScanSettings {
            rmin_auto: true,
            ..crate::test_archive::archive().settings
        }
    }

    #[test]
    fn the_ends_are_included_with_n_points() {
        let sweep = Sweep {
            variable: Variable::Salt,
            from: 10.0,
            to: 100.0,
            n: 4,
        };
        assert_eq!(
            sweep.values().collect::<Vec<_>>(),
            [10.0, 40.0, 70.0, 100.0]
        );
    }

    /// Salt is entered in mM; scans take mol/L.
    #[test]
    fn only_the_swept_variable_changes() {
        let at = |variable, x| {
            let sweep = Sweep {
                variable,
                from: x,
                to: x + 1.0,
                n: 2,
            };
            SweepPoints::new(sweep, base()).next().unwrap()
        };
        let (x, salt) = at(Variable::Salt, 20.0);
        assert_eq!(x, 20.0);
        assert_eq!(
            salt,
            ScanSettings {
                ionic_strength: 0.02,
                ..base()
            }
        );
        let (_, ph) = at(Variable::Ph, 4.0);
        assert_eq!(ph, ScanSettings { ph: 4.0, ..base() });
        let (_, t) = at(Variable::Temperature, 310.0);
        assert_eq!(
            t,
            ScanSettings {
                temperature: 310.0,
                ..base()
            }
        );
    }

    /// One R_min for every point, so the hard-core split is the same along the curve.
    #[test]
    fn rmin_is_pinned_from_the_second_point_on() {
        let sweep = Sweep {
            variable: Variable::Ph,
            from: 4.0,
            to: 5.0,
            n: 2,
        };
        let mut points = SweepPoints::new(sweep, base());
        let mut first = crate::test_archive::archive();
        first.settings.rmin = 6.5;
        assert!(points.next().unwrap().1.rmin_auto);
        points.push(&first).unwrap();
        let (x, second) = points.next().unwrap();
        assert_eq!(x, 5.0);
        assert_eq!((second.rmin, second.rmin_auto), (6.5, false));
        points.push(&first).unwrap();
        assert!(points.next().is_none());
    }

    fn ph_sweep() -> SweepPoints {
        let sweep = Sweep {
            variable: Variable::Ph,
            from: 4.0,
            to: 9.0,
            n: 6,
        };
        SweepPoints::new(sweep, base())
    }

    #[test]
    fn a_point_keeps_its_virial_conditions_and_moments() {
        let mut archive = crate::test_archive::archive();
        archive.bodies[0].multipole = Some(crate::test_archive::multipole());
        let mut points = ph_sweep();
        points.push(&archive).unwrap();
        let point = &points.points()[0];
        assert_eq!(point.x, 4.0);
        assert_eq!(point.facts.virial.b2(), archive.virial.b2());
        assert_eq!(point.facts.ph, archive.settings.ph);
        assert_eq!(point.facts.ionic_strength, archive.settings.ionic_strength);
        let a = &point.facts.bodies[0];
        assert_eq!((a.charge, a.capacitance), (7.5, Some(1.75)));
    }

    /// Bead radii don't follow pH, salt or T, so a moved reference means
    /// the points aren't comparable.
    #[test]
    fn a_changed_steric_reference_is_refused() {
        let first = crate::test_archive::archive();
        let mut moved = first.clone();
        moved.virial = moved
            .virial
            .with_reference(first.virial.reference() * 1.001);
        let mut points = ph_sweep();
        points.push(&first).unwrap();
        let error = points.push(&moved).unwrap_err().to_string();
        assert!(error.contains("reference"), "{error}");
        assert_eq!(points.points().len(), 1);
    }

    /// The pane and the plot read one Summary; B₂ʰˢ is display only, so it
    /// moves B₂* and nothing the scan computed.
    #[test]
    fn the_pane_and_the_plot_agree_and_the_normaliser_moves_only_b2_star() {
        use crate::twobody::reference::Normaliser;
        let archive = crate::test_archive::associating();
        let reference = archive.virial.reference();
        let custom = Normaliser::Custom { b2_hs: 1.0e5 };
        let mut points = ph_sweep();
        points.push(&archive).unwrap();

        let pane = Facts::of(&archive).summary(&custom);
        let plotted = curve(&points.summaries(&custom), Key::B2Star);
        assert_eq!(plotted, [[4.0, pane.value(Key::B2Star).unwrap()]]);
        assert_eq!(pane.value(Key::B2Star), Some(archive.virial.b2() / 1.0e5));

        let steric = Facts::of(&archive).summary(&Normaliser::Steric);
        assert_eq!(pane.value(Key::Kd), steric.value(Key::Kd));
        assert!(pane.value(Key::Kd).is_some());
        assert_eq!(archive.virial.reference(), reference);
    }

    /// Header cell index, then each row's cells.
    fn table(csv: &str) -> (Vec<&str>, Vec<Vec<&str>>) {
        let mut lines = csv.lines().skip(1);
        let header = lines.next().unwrap().split(',').collect();
        (header, lines.map(|l| l.split(',').collect()).collect())
    }

    fn column<'a>(header: &[&str], rows: &'a [Vec<&str>], name: &str) -> Vec<&'a str> {
        let i = header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("{name}"));
        rows.iter().map(|r| r[i]).collect()
    }

    #[test]
    fn the_csv_names_its_normaliser_rmin_and_pair_first() {
        let mut points = ph_sweep();
        points.push(&crate::test_archive::associating()).unwrap();
        let custom = Normaliser::Custom { b2_hs: 1.0e5 };
        let csv = points.csv(&custom, ["lys.pdb", "lys.pdb"]).unwrap();
        let first = csv.lines().next().unwrap();
        assert!(first.starts_with("# B₂ʰˢ = 1.000e5 Å³ (custom)"), "{first}");
        assert!(first.contains("R_min = 5.0 Å"), "{first}");
        assert!(first.contains("model test"), "{first}");
        assert!(first.contains("pair lys.pdb/lys.pdb"), "{first}");
    }

    /// Cells round-trip to the Summary's numbers; empty where it has none.
    #[test]
    fn csv_cells_are_the_summarys_numbers_with_units_in_the_header() {
        let mut points = ph_sweep();
        let at = |ph| {
            let mut archive = crate::test_archive::associating();
            archive.settings.ph = ph;
            archive
        };
        let archive = at(4.0);
        points.push(&archive).unwrap();
        points.push(&crate::test_archive::archive()).unwrap_err(); // other reference
        points.push(&at(5.0)).unwrap();
        let steric = Normaliser::Steric;
        let csv = points.csv(&steric, ["a", "b"]).unwrap();
        let (header, rows) = table(&csv);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.len() == header.len()));
        let summary = Facts::of(&archive).summary(&steric);
        let b2: Vec<f64> = column(&header, &rows, "B2/Å³")
            .iter()
            .map(|c| c.parse().unwrap())
            .collect();
        assert_eq!(b2, [summary.value(Key::B2).unwrap(); 2]);
        assert_eq!(column(&header, &rows, "pH"), ["4", "5"]);
        let kd: f64 = column(&header, &rows, "Kd/M")[0].parse().unwrap();
        assert_eq!(Some(kd), summary.value(Key::Kd));
        assert_eq!(column(&header, &rows, "CA/e²"), ["", ""]);
    }

    #[test]
    fn a_heterodimer_has_its_own_a_and_b_columns() {
        let mut archive = crate::test_archive::associating();
        archive.settings.homo_dimer = false;
        archive.bodies[1].multipole = Some(crate::test_archive::multipole());
        let mut points = ph_sweep();
        points.push(&archive).unwrap();
        let csv = points.csv(&Normaliser::Steric, ["a", "b"]).unwrap();
        let (header, rows) = table(&csv);
        assert_eq!(column(&header, &rows, "ZA/e"), ["0"]);
        assert_eq!(column(&header, &rows, "ZB/e"), ["7.5"]);
        assert_eq!(column(&header, &rows, "muB/eÅ"), ["40"]);
    }
}
