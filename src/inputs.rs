use crate::conditions::SolutionConditions;
use duello::web_api::ScanSettings;
use duello::Subdivision;

/// Force field model choices.
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CgPolicy {
    Single,
    Multi,
}

/// Twobody's own scan parameters; the solution comes from
/// [`SolutionConditions`], and a homo-dimer from picking one structure twice.
#[derive(Clone)]
pub struct ScanInputs {
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub max_ndiv: usize,
    pub subdivision: Subdivision,
    pub gradient_threshold: f64,
    pub cutoff: f64,
    pub grid_size: usize,
    pub model: Model,
    pub cg: CgPolicy,
}

impl ScanInputs {
    /// `homo_dimer` is left false: only the structures can tell.
    pub fn settings(&self, conditions: &SolutionConditions) -> ScanSettings {
        ScanSettings {
            ph: conditions.ph,
            model: self.model.name().to_string(),
            single_bead: self.cg == CgPolicy::Single,
            rmin: self.rmin,
            rmax: self.rmax,
            dr: self.dr,
            ionic_strength: conditions.ionic_strength,
            temperature: conditions.temperature,
            max_ndiv: self.max_ndiv,
            gradient_threshold: self.gradient_threshold,
            cutoff: self.cutoff,
            grid_size: self.grid_size,
            homo_dimer: false,
            subdivision: self.subdivision,
        }
    }

    /// Inputs matching an imported scan, so the panel describes what's shown;
    /// [`SolutionConditions::adopt`] takes the rest.
    /// A model the UI can't select leaves `fallback` chosen; callers warn.
    pub fn from_settings(s: &ScanSettings, fallback: Model) -> Self {
        Self {
            rmin: s.rmin,
            rmax: s.rmax,
            dr: s.dr,
            max_ndiv: s.max_ndiv,
            subdivision: s.subdivision,
            gradient_threshold: s.gradient_threshold,
            cutoff: s.cutoff,
            grid_size: s.grid_size,
            model: Model::from_name(&s.model).unwrap_or(fallback),
            cg: if s.single_bead {
                CgPolicy::Single
            } else {
                CgPolicy::Multi
            },
        }
    }
}

impl Default for ScanInputs {
    fn default() -> Self {
        Self {
            rmin: 26.0,
            rmax: 60.0,
            dr: 1.0,
            max_ndiv: 2,
            subdivision: Subdivision::default(),
            gradient_threshold: 0.5,
            cutoff: 20.0,
            grid_size: 200,
            model: Model::Calvados3,
            cg: CgPolicy::Single,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditions::SolutionConditions;

    fn conditions() -> SolutionConditions {
        let mut conditions = SolutionConditions::default();
        conditions.temperature = 310.0;
        conditions.ph = 4.5;
        conditions.ionic_strength = 0.02;
        conditions
    }

    #[test]
    fn model_names_from_archives_map_back_to_ui_models() {
        assert_eq!(Model::from_name("kimhummer"), Some(Model::KimHummer));
        assert_eq!(Model::from_name("Calvados3"), Some(Model::Calvados3));
        assert_eq!(Model::from_name("custom topology"), None);
    }

    /// Onebody owns the solution, so the scan takes it from there.
    #[test]
    fn settings_take_the_solution_from_the_conditions() {
        let conditions = conditions();
        let settings = ScanInputs::default().settings(&conditions);
        assert_eq!(
            (settings.temperature, settings.ph, settings.ionic_strength),
            (310.0, 4.5, 0.02)
        );
        assert_eq!(
            (settings.rmin, settings.rmax, settings.dr),
            (26.0, 60.0, 1.0)
        );
    }

    #[test]
    fn imported_settings_restore_the_inputs_and_conditions_they_came_from() {
        let inputs = ScanInputs {
            model: Model::Pasquier,
            cg: CgPolicy::Multi,
            subdivision: Subdivision::Lattice,
            ..Default::default()
        };
        let conditions = conditions();
        let settings = inputs.settings(&conditions);
        let restored = ScanInputs::from_settings(&settings, Model::Calvados3);
        assert_eq!(
            (restored.model, restored.cg, restored.subdivision),
            (Model::Pasquier, CgPolicy::Multi, Subdivision::Lattice)
        );
        let mut adopted = SolutionConditions::default();
        adopted.adopt(&settings);
        assert_eq!(adopted, conditions);
    }

    /// The panel must still show the scan's numbers when its model isn't selectable.
    #[test]
    fn unknown_models_keep_the_current_choice_but_restore_settings() {
        let mut settings = ScanInputs::default().settings(&Default::default());
        settings.model = "custom topology".into();
        settings.rmax = 80.0;
        let restored = ScanInputs::from_settings(&settings, Model::KimHummer);
        assert_eq!(restored.model, Model::KimHummer);
        assert_eq!(restored.rmax, 80.0);
    }
}
