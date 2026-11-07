//! Model and coarse-graining policy, set in Onebody: they decide titration, bead
//! properties and the pair energy, so every tab must see the same choice.

use crate::widgets::{grid, pick, row};
use duello::web_api::ScanSettings;

const MODEL_TIP: &str =
    "Residue-level force field for bead properties and the short-range pair energy";
const CG_TIP: &str = "One bead per residue, or several per residue";

/// Force field model choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Model {
    Calvados3,
    KimHummer,
    Pasquier,
}

impl Model {
    /// Every selectable model; the picker and name lookup share this list.
    pub const ALL: [Model; 3] = [Model::Calvados3, Model::KimHummer, Model::Pasquier];

    pub fn name(&self) -> &str {
        match self {
            Model::Calvados3 => "calvados3",
            Model::KimHummer => "kimhummer",
            Model::Pasquier => "pasquier",
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Model::Calvados3 => "CALVADOS 3",
            Model::KimHummer => "Kim-Hummer",
            Model::Pasquier => "Pasquier",
        }
    }

    /// `None` for models the UI can't select, e.g. "custom topology" from a CLI
    /// scan with `--top`. Case-insensitive, as CLI spellings may vary.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|model| model.name().eq_ignore_ascii_case(name))
    }
}

/// Coarse-graining policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CgPolicy {
    Single,
    Multi,
}

impl CgPolicy {
    const ALL: [CgPolicy; 2] = [CgPolicy::Single, CgPolicy::Multi];

    fn label(self) -> &'static str {
        match self {
            CgPolicy::Single => "Single bead",
            CgPolicy::Multi => "Multi bead",
        }
    }
}

impl From<bool> for CgPolicy {
    /// From an archive's `single_bead`.
    fn from(single_bead: bool) -> Self {
        if single_bead {
            CgPolicy::Single
        } else {
            CgPolicy::Multi
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForceField {
    pub model: Model,
    pub cg: CgPolicy,
}

impl Default for ForceField {
    fn default() -> Self {
        Self {
            model: Model::Calvados3,
            cg: CgPolicy::Single,
        }
    }
}

impl ForceField {
    pub fn single_bead(&self) -> bool {
        self.cg == CgPolicy::Single
    }

    /// A scan's choice; `None` for a model the UI can't select.
    fn of_scan(settings: &ScanSettings) -> Option<Self> {
        Some(Self {
            model: Model::from_name(&settings.model)?,
            cg: settings.single_bead.into(),
        })
    }

    /// Takes an imported scan's choice. One the UI can't select (e.g. a CLI `--top`
    /// scan) keeps the current choice, and the returned warning says so.
    pub fn adopt(&mut self, settings: &ScanSettings) -> Option<String> {
        match Self::of_scan(settings) {
            Some(scan) => {
                *self = scan;
                None
            }
            None => Some(format!(
                "Imported scan uses model '{}', which the panel can't select",
                settings.model
            )),
        }
    }

    /// Why a scan no longer reflects this choice, if it doesn't.
    pub fn stale(&self, settings: &ScanSettings) -> Option<String> {
        let Some(scan) = Self::of_scan(settings) else {
            return Some(format!("the scan used model '{}'", settings.model));
        };
        let differs =
            |was: &str, now: &str| (was != now).then(|| format!("{was} in the scan, {now} now"));
        let reasons: Vec<_> = [
            differs(scan.model.label(), self.model.label()).map(|r| format!("model {r}")),
            differs(scan.cg.label(), self.cg.label()),
        ]
        .into_iter()
        .flatten()
        .collect();
        (!reasons.is_empty()).then(|| reasons.join("; "))
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        grid("forcefield").show(ui, |ui| {
            row(ui, "Model", MODEL_TIP, |ui| {
                pick(
                    ui,
                    "model_combo",
                    &mut self.model,
                    &Model::ALL,
                    Model::label,
                    None,
                )
            });
            row(ui, "CG policy", CG_TIP, |ui| {
                pick(
                    ui,
                    "cg_combo",
                    &mut self.cg,
                    &CgPolicy::ALL,
                    |cg| cg.label(),
                    None,
                )
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditions::SolutionConditions;
    use crate::inputs::ScanInputs;

    fn settings(forcefield: &ForceField) -> ScanSettings {
        ScanInputs::default().settings(&SolutionConditions::default(), forcefield)
    }

    #[test]
    fn model_names_from_archives_map_back_to_ui_models() {
        assert_eq!(Model::from_name("kimhummer"), Some(Model::KimHummer));
        assert_eq!(Model::from_name("Calvados3"), Some(Model::Calvados3));
        assert_eq!(Model::from_name("custom topology"), None);
    }

    #[test]
    fn imported_scans_restore_the_force_field() {
        let original = ForceField {
            model: Model::Pasquier,
            cg: CgPolicy::Multi,
        };
        let mut adopted = ForceField::default();
        assert_eq!(adopted.adopt(&settings(&original)), None);
        assert_eq!(adopted, original);
    }

    /// A model the UI can't select (e.g. a CLI `--top` scan) says nothing about the
    /// panel's policy either, so the whole choice stays.
    #[test]
    fn unknown_models_keep_the_current_choice_and_warn() {
        let mut imported = settings(&ForceField {
            model: Model::Calvados3,
            cg: CgPolicy::Multi,
        });
        imported.model = "custom topology".into();
        let mut current = ForceField {
            model: Model::KimHummer,
            cg: CgPolicy::Single,
        };
        let warning = current.adopt(&imported).unwrap();
        assert!(warning.contains("custom topology"), "{warning}");
        assert_eq!(current.model, Model::KimHummer);
        assert_eq!(current.cg, CgPolicy::Single);
    }

    /// Twobody shows the scan beside Onebody's current choice, which may have moved on.
    #[test]
    fn stale_names_what_differs_from_the_scan() {
        let scan = settings(&ForceField {
            model: Model::Pasquier,
            cg: CgPolicy::Multi,
        });
        let why = ForceField::default().stale(&scan).unwrap();
        assert!(
            why.contains("Pasquier") && why.contains("CALVADOS 3"),
            "{why}"
        );
        assert!(
            why.contains("Multi bead") && why.contains("Single bead"),
            "{why}"
        );
        let same = ForceField {
            model: Model::Pasquier,
            cg: CgPolicy::Multi,
        };
        assert_eq!(same.stale(&scan), None);
    }
}
