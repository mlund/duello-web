//! The selected structure, coarse-grained and titrated under Onebody's force
//! field and conditions, for the analysis windows. Titration steps a little each
//! frame and starts over when anything it depends on changes.

use super::scene::{self, Colouring};
use super::table::{BeadTable, Column};
use super::{panes, Pane};
use crate::budget::{self, Instant};
use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use crate::structures::StructureId;
use crate::viewer::Viewer;
use duello::archive::{CgBody, StructureFile};
use duello::molecule::{titration_curve, CurvePoint, Molecule, MoleculeSpec, Step, Titrating};
use duello::system::TITRATION_SWEEPS;
use std::hash::{Hash, Hasher};

/// Fixed, so the same inputs always titrate to the same charges.
const SEED: u64 = 1;

/// MC sweeps in the first frame after a restart; `budget::retune` takes it from there.
const FIRST_SWEEPS: u64 = 50;

/// Titration curve abscissa: pH 2–12 in steps of 0.1.
fn ph_grid() -> Vec<f64> {
    (0..=100).map(|i| 2.0 + 0.1 * f64::from(i)).collect()
}

/// Everything the titration depends on. Contents rather than the id or name, so
/// renaming, or reopening a session, doesn't start over.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    contents: u64,
    forcefield: ForceField,
    ph: f64,
    ionic_strength: f64,
    temperature: f64,
}

impl Key {
    /// The curve spans pH, so doesn't depend on it.
    fn curve(self) -> Self {
        Self { ph: 0.0, ..self }
    }

    /// Same molecule, perhaps other conditions: its last result still describes it.
    fn same_molecule(self, other: Self) -> bool {
        (self.contents, self.forcefield) == (other.contents, other.forcefield)
    }
}

pub struct MoleculeView {
    key: Option<Key>,
    titrating: Option<Titrating>,
    /// Last finished titration of this molecule, shown while conditions re-titrate.
    shown: Option<Box<Molecule>>,
    /// Counts molecules shown, naming them for the viewer and tables.
    generation: u64,
    error: Option<String>,
    /// HH net charge over [`ph_grid`], for `curve_for`.
    curve: Vec<CurvePoint>,
    curve_for: Option<Key>,
    sweeps_per_frame: u64,
    /// Contents hash of the last structure seen; an id's contents never change.
    hashed: Option<(StructureId, u64)>,
    viewer: Viewer,
    colouring: Colouring,
    atoms: BeadTable,
    sites: BeadTable,
}

const ATOM_COLUMNS: &[Column] = &[
    Column::Index,
    Column::Chain,
    Column::Residue,
    Column::Kind,
    Column::Charge,
    Column::Mass,
    Column::Sigma,
    Column::Epsilon,
    Column::Lambda,
];

const SITE_COLUMNS: &[Column] = &[
    Column::Chain,
    Column::Residue,
    Column::Kind,
    Column::Charge,
    Column::Variance,
];

impl Default for MoleculeView {
    fn default() -> Self {
        Self {
            key: None,
            titrating: None,
            shown: None,
            generation: 0,
            error: None,
            curve: Vec::new(),
            curve_for: None,
            sweeps_per_frame: FIRST_SWEEPS,
            hashed: None,
            viewer: Viewer::default(),
            colouring: Colouring::Charge,
            atoms: BeadTable::new("atoms_table", ATOM_COLUMNS, |_| true),
            sites: BeadTable::new("sites_table", SITE_COLUMNS, |b| b.charge_variance.is_some()),
        }
    }
}

impl MoleculeView {
    /// Call every frame Onebody shows; true while titrating, so the caller keeps
    /// repainting.
    pub fn tick(
        &mut self,
        structure: Option<(StructureId, &StructureFile)>,
        forcefield: &ForceField,
        conditions: &SolutionConditions,
    ) -> bool {
        let Some((id, file)) = structure else {
            // Only the list emptying gets here; UI choices stay.
            (self.key, self.titrating, self.shown, self.error) = (None, None, None, None);
            return false;
        };
        let key = Key {
            contents: self.contents_hash(id, &file.contents),
            forcefield: *forcefield,
            ph: conditions.ph,
            ionic_strength: conditions.ionic_strength,
            temperature: conditions.temperature,
        };
        if self.key != Some(key) {
            self.restart(key, &file.contents);
        }
        self.advance();
        self.titrating.is_some()
    }

    /// The titrated body; while conditions re-titrate, the last one.
    pub fn body(&self) -> Option<&CgBody> {
        self.shown.as_ref().map(|m| &m.body)
    }

    pub fn ui_pane(&mut self, ui: &mut egui::Ui, pane: Pane) {
        // While re-titrating, progress shows once, in the side panel.
        let (Some(molecule), Some(key)) = (&self.shown, self.key) else {
            return self.status_ui(ui);
        };
        let generation = self.generation;
        match pane {
            Pane::Multipole => panes::multipole(ui, molecule),
            Pane::Titration => {
                let height = 0.5 * ui.available_height();
                panes::titration_plot(ui, &self.curve, key.ph, height);
                self.sites.ui(ui, &molecule.beads, generation);
            }
            Pane::Viewer => {
                colouring_picker(ui, &mut self.colouring);
                let colouring = self.colouring;
                self.viewer.ui(ui, (generation, colouring), || {
                    scene::scene(&molecule.body, colouring)
                });
            }
            Pane::Atoms => self.atoms.ui(ui, &molecule.beads, generation),
        }
    }

    /// Progress or failure; nothing once the molecule is ready.
    pub fn status_ui(&self, ui: &mut egui::Ui) {
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, e);
        } else if let Some(titrating) = &self.titrating {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.add(
                    egui::ProgressBar::new(titrating.progress() as f32)
                        .show_percentage()
                        .text("Titrating…"),
                )
                .on_hover_text(
                    "Monte Carlo titration with as many sweeps as a scan; the windows \
                     show the last result until it finishes",
                );
            });
        } else if self.shown.is_none() {
            ui.weak("Load a structure to analyse it.");
        }
    }

    fn contents_hash(&mut self, id: StructureId, contents: &[u8]) -> u64 {
        match self.hashed {
            Some((seen, hash)) if seen == id => hash,
            _ => {
                let mut hasher = std::hash::DefaultHasher::new();
                contents.hash(&mut hasher);
                let hash = hasher.finish();
                self.hashed = Some((id, hash));
                hash
            }
        }
    }

    fn restart(&mut self, key: Key, pdb: &[u8]) {
        if !self.key.is_some_and(|old| old.same_molecule(key)) {
            self.shown = None;
        }
        self.key = Some(key);
        self.sweeps_per_frame = FIRST_SWEEPS;
        let spec = MoleculeSpec {
            model: key.forcefield.model.name().to_string(),
            single_bead: key.forcefield.single_bead(),
            ph: key.ph,
            ionic_strength: key.ionic_strength,
            temperature: key.temperature,
            mc_sweeps: TITRATION_SWEEPS,
        };
        let started = self
            .update_curve(key, pdb, &spec)
            .and_then(|()| Titrating::start(pdb, spec, SEED));
        (self.titrating, self.error) = match started {
            Ok(titrating) => (Some(titrating), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };
    }

    fn update_curve(&mut self, key: Key, pdb: &[u8], spec: &MoleculeSpec) -> anyhow::Result<()> {
        if self.curve_for != Some(key.curve()) {
            self.curve_for = None;
            self.curve = titration_curve(pdb, spec, &ph_grid())?;
            self.curve_for = Some(key.curve());
        }
        Ok(())
    }

    /// One frame's sweeps.
    fn advance(&mut self) {
        let Some(titrating) = self.titrating.take() else {
            return;
        };
        let start = Instant::now();
        let step = titrating.advance(self.sweeps_per_frame as usize);
        self.sweeps_per_frame = budget::retune(self.sweeps_per_frame, start);
        match step {
            Ok(Step::Running(titrating)) => self.titrating = Some(titrating),
            Ok(Step::Done(molecule)) => {
                self.shown = Some(molecule);
                self.generation += 1;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }
}

fn colouring_picker(ui: &mut egui::Ui, colouring: &mut Colouring) {
    ui.horizontal(|ui| {
        let tip = |c: Colouring| match c {
            Colouring::Charge => "Red negative, grey neutral, blue positive; full at ±1 e",
            Colouring::Kind => "One colour per atom kind",
        };
        ui.label("Colour")
            .on_hover_text("What the bead colours show");
        for c in Colouring::ALL {
            ui.selectable_value(colouring, c, c.label())
                .on_hover_text(tip(c));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lys–Asp–Ala, a few heavy atoms each, 3.8 Å apart.
    fn tripeptide() -> Vec<u8> {
        let residues: [(&str, &[&str]); 3] = [
            ("LYS", &["N", "CA", "C", "O", "CB", "NZ"]),
            ("ASP", &["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]),
            ("ALA", &["N", "CA", "C", "O", "CB", "OXT"]),
        ];
        let mut pdb = String::new();
        let mut serial = 1;
        for (seq, (name, atoms)) in residues.iter().enumerate() {
            for (k, atom) in atoms.iter().enumerate() {
                let (x, y) = (3.8 * seq as f64, 0.5 * k as f64);
                let element = &atom[..1];
                pdb += &format!(
                    "ATOM  {serial:5} {atom:<4} {name} A{:4}    {x:8.3}{y:8.3}{:8.3}  1.00  0.00          {element:>2}\n",
                    seq + 1,
                    0.0
                );
                serial += 1;
            }
        }
        pdb += "END\n";
        pdb.into_bytes()
    }

    fn file(name: &str) -> StructureFile {
        StructureFile {
            name: name.into(),
            contents: tripeptide(),
        }
    }

    fn net_charge(view: &MoleculeView) -> f64 {
        view.body().unwrap().charges.iter().sum()
    }

    /// Ticks until the titration finishes.
    fn finish(view: &mut MoleculeView, file: &StructureFile, conditions: &SolutionConditions) {
        let id = StructureId::for_tests(0);
        for _ in 0..10_000 {
            if !view.tick(Some((id, file)), &ForceField::default(), conditions) {
                return;
            }
        }
        panic!("titration never finished");
    }

    /// Renaming changes nothing the titration sees; pH does.
    #[test]
    fn ph_restarts_the_titration_but_renaming_does_not() {
        let mut view = MoleculeView::default();
        let neutral = SolutionConditions::default();
        finish(&mut view, &file("a"), &neutral);
        let at_7 = net_charge(&view);

        let id = StructureId::for_tests(0);
        let ff = ForceField::default();
        assert!(!view.tick(Some((id, &file("renamed"))), &ff, &neutral));
        assert_eq!(net_charge(&view), at_7);

        let mut acidic = SolutionConditions::default();
        acidic.ph = 3.0;
        assert!(view.tick(Some((id, &file("renamed"))), &ff, &acidic));
        assert_eq!(net_charge(&view), at_7, "shown until pH 3 finishes");
        finish(&mut view, &file("renamed"), &acidic);
        // Asp and the C-terminus take protons: from about 0 to about +1.
        assert!(
            net_charge(&view) - at_7 > 0.8,
            "{at_7} → {}",
            net_charge(&view)
        );
    }
}
