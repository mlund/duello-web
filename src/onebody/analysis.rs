//! The selected structure, coarse-grained and titrated under Onebody's force
//! field and conditions, for the analysis windows. Titration steps a little each
//! frame and starts over when anything it depends on changes.

use super::panes::{self, Quantity};
use super::table::BeadTable;
use super::view3d::MoleculeScene;
use super::Pane;
use crate::budget::{self, Instant};
use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use crate::structures::StructureId;
use duello::archive::{CgBody, StructureFile};
use duello::molecule::{Curve, Molecule, MoleculeSpec, Step, Titrating};
use duello::system::TITRATION_SWEEPS;
use std::hash::{Hash, Hasher};

/// Fixed, so the same inputs always titrate to the same charges.
const SEED: u64 = 1;

/// MC sweeps in the first frame after a restart; `budget::retune` takes it from there.
const FIRST_SWEEPS: u64 = 50;

/// pH 2–12 in `steps` intervals.
fn ph_grid(steps: u32) -> Vec<f64> {
    (0..=steps)
        .map(|i| 2.0 + 10.0 * f64::from(i) / f64::from(steps))
        .collect()
}

/// Henderson–Hasselbalch is cheap: every 0.1 pH.
const IDEAL_STEPS: u32 = 100;

/// MC costs a titration per point (~0.25 s natively for 2CDS): every 0.5 pH.
/// Points use a scan's sweeps, so they agree with Multipole at their pH.
const MC_CURVE_STEPS: u32 = 20;

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
    /// Last finished titration of this molecule and what it was titrated
    /// under, shown while conditions re-titrate.
    shown: Option<(Key, Box<Molecule>)>,
    /// Counts molecules shown, naming them for the viewer and tables.
    generation: u64,
    error: Option<String>,
    /// MC and ideal curves for a [`Key::curve`]; MC fills in once the
    /// current-pH titration is done.
    curve: Option<(Key, Curve)>,
    quantity: Quantity,
    sweeps_per_frame: u64,
    /// Contents hash of the last structure seen; an id's contents never change.
    hashed: Option<(StructureId, u64)>,
    scene: MoleculeScene,
    atoms: BeadTable,
}

impl Default for MoleculeView {
    fn default() -> Self {
        Self {
            key: None,
            titrating: None,
            shown: None,
            generation: 0,
            error: None,
            curve: None,
            quantity: Quantity::Charge,
            sweeps_per_frame: FIRST_SWEEPS,
            hashed: None,
            scene: MoleculeScene::default(),
            atoms: BeadTable::default(),
        }
    }
}

impl MoleculeView {
    /// Call every frame Onebody shows; true while titrating, filling the curve
    /// or holding a change, so the caller keeps repainting. Changes wait until
    /// `settled` (nothing dragged), so a dragged value restarts once.
    pub fn tick(
        &mut self,
        structure: Option<(StructureId, &StructureFile)>,
        forcefield: &ForceField,
        conditions: &SolutionConditions,
        settled: bool,
    ) -> bool {
        let Some((id, file)) = structure else {
            // Only the list emptying gets here; UI choices stay.
            (self.key, self.titrating, self.shown, self.error) = (None, None, None, None);
            self.curve = None;
            return false;
        };
        let key = Key {
            contents: self.contents_hash(id, &file.contents),
            forcefield: *forcefield,
            ph: conditions.ph,
            ionic_strength: conditions.ionic_strength,
            temperature: conditions.temperature,
        };
        let held = self.key != Some(key) && !settled;
        if self.key != Some(key) && settled {
            self.restart(key, &file.contents);
        }
        self.advance();
        if held {
            return true;
        }
        self.titrating.is_some() || self.curve_running()
    }

    fn curve_running(&self) -> bool {
        self.curve.as_ref().is_some_and(|(_, c)| !c.is_done())
    }

    /// The titrated body; while conditions re-titrate, the last one.
    pub fn body(&self) -> Option<&CgBody> {
        self.shown.as_ref().map(|(_, m)| &m.body)
    }

    pub fn ui_pane(&mut self, ui: &mut egui::Ui, pane: Pane) {
        // While re-titrating, progress shows once, in the side panel.
        let (Some((titrated, molecule)), Some(key)) = (&self.shown, self.key) else {
            return self.status_ui(ui);
        };
        let generation = self.generation;
        match pane {
            Pane::Multipole => panes::multipole(ui, molecule),
            Pane::Titration => {
                let (ideal, mc) = self
                    .curve
                    .as_ref()
                    .map_or((&[][..], &[][..]), |(_, c)| (c.ideal(), c.points()));
                panes::titration(ui, &mut self.quantity, ideal, mc, key.ph);
            }
            // Screened as titrated, so charges and screening agree.
            Pane::Viewer => {
                let (t, i) = (titrated.temperature, titrated.ionic_strength);
                self.scene.ui(ui, molecule, generation, t, i);
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
        if let Some((_, curve)) = self.curve.as_ref().filter(|(_, c)| !c.is_done()) {
            ui.add(
                egui::ProgressBar::new(curve.progress() as f32)
                    .show_percentage()
                    .text("Titration curve…"),
            )
            .on_hover_text(format!(
                "Monte Carlo titration every {} pH, as many sweeps as a scan",
                10.0 / f64::from(MC_CURVE_STEPS)
            ));
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
            self.scene.reset_camera();
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
            Err(e) => {
                // A molecule that can't titrate gets no curve either.
                self.curve = None;
                (None, Some(format!("{e:#}")))
            }
        };
    }

    fn update_curve(&mut self, key: Key, pdb: &[u8], spec: &MoleculeSpec) -> anyhow::Result<()> {
        if self.curve.as_ref().is_some_and(|(k, _)| *k == key.curve()) {
            return Ok(());
        }
        self.curve = None;
        let (mc, ideal) = (ph_grid(MC_CURVE_STEPS), ph_grid(IDEAL_STEPS));
        self.curve = Some((key.curve(), Curve::start(pdb, spec, &mc, &ideal, SEED)?));
        Ok(())
    }

    /// One frame's sweeps: the current pH first, then the curve. Both titrate the
    /// same molecule, so they share one budget.
    fn advance(&mut self) {
        if let Some((_, curve)) = &mut self.curve {
            curve.advance_ideal();
        }
        let sweeps = self.sweeps_per_frame as usize;
        let start = Instant::now();
        if let Some(titrating) = self.titrating.take() {
            match titrating.advance(sweeps) {
                Ok(Step::Running(titrating)) => self.titrating = Some(titrating),
                Ok(Step::Done(molecule)) => {
                    let key = self.key.expect("titrations start with a key");
                    self.shown = Some((key, molecule));
                    self.generation += 1;
                }
                Err(e) => self.error = Some(format!("{e:#}")),
            }
        } else if let Some((_, curve)) = self.curve.as_mut().filter(|(_, c)| !c.is_done()) {
            curve.advance(sweeps);
        } else {
            return;
        }
        self.sweeps_per_frame = budget::retune(self.sweeps_per_frame, start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onebody::tests::tripeptide;

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
            if !view.tick(Some((id, file)), &ForceField::default(), conditions, true) {
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
        assert!(!view.tick(Some((id, &file("renamed"))), &ff, &neutral, true));
        assert_eq!(net_charge(&view), at_7);

        let mut acidic = SolutionConditions::default();
        acidic.ph = 3.0;
        assert!(view.tick(Some((id, &file("renamed"))), &ff, &acidic, true));
        assert_eq!(net_charge(&view), at_7, "shown until pH 3 finishes");
        finish(&mut view, &file("renamed"), &acidic);
        // Asp and the C-terminus take protons: from about 0 to about +1.
        assert!(
            net_charge(&view) - at_7 > 0.8,
            "{at_7} → {}",
            net_charge(&view)
        );
    }

    /// Dragging a value would otherwise restart every frame; the change waits
    /// for the release.
    #[test]
    fn restarts_wait_until_a_drag_ends() {
        let mut view = MoleculeView::default();
        finish(&mut view, &file("a"), &SolutionConditions::default());
        let at_7 = net_charge(&view);
        let mut acidic = SolutionConditions::default();
        acidic.ph = 3.0;
        let (id, ff) = (StructureId::for_tests(0), ForceField::default());
        for _ in 0..1000 {
            assert!(view.tick(Some((id, &file("a"))), &ff, &acidic, false));
        }
        assert_eq!(net_charge(&view), at_7, "held while dragging");
        finish(&mut view, &file("a"), &acidic);
        assert!(net_charge(&view) - at_7 > 0.8);
    }
}
