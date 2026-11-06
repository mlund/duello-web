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

/// Scan parameters editable in the UI.
#[derive(Clone)]
pub struct ScanInputs {
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub molarity: f64,
    pub temperature: f64,
    pub max_ndiv: usize,
    pub subdivision: Subdivision,
    pub gradient_threshold: f64,
    pub cutoff: f64,
    pub grid_size: usize,
    pub ph: f64,
    pub model: Model,
    pub cg: CgPolicy,
    pub homo_dimer: bool,
}

impl From<&ScanInputs> for ScanSettings {
    fn from(inputs: &ScanInputs) -> Self {
        Self {
            ph: inputs.ph,
            model: inputs.model.name().to_string(),
            single_bead: inputs.cg == CgPolicy::Single,
            rmin: inputs.rmin,
            rmax: inputs.rmax,
            dr: inputs.dr,
            molarity: inputs.molarity,
            temperature: inputs.temperature,
            max_ndiv: inputs.max_ndiv,
            gradient_threshold: inputs.gradient_threshold,
            cutoff: inputs.cutoff,
            grid_size: inputs.grid_size,
            homo_dimer: inputs.homo_dimer,
            subdivision: inputs.subdivision,
        }
    }
}

impl ScanInputs {
    /// Inputs matching an imported scan, so the panel describes what's shown.
    /// A model the UI can't select leaves `fallback` chosen; callers warn.
    pub fn from_settings(s: &ScanSettings, fallback: Model) -> Self {
        Self {
            rmin: s.rmin,
            rmax: s.rmax,
            dr: s.dr,
            molarity: s.molarity,
            temperature: s.temperature,
            max_ndiv: s.max_ndiv,
            subdivision: s.subdivision,
            gradient_threshold: s.gradient_threshold,
            cutoff: s.cutoff,
            grid_size: s.grid_size,
            ph: s.ph,
            model: Model::from_name(&s.model).unwrap_or(fallback),
            cg: if s.single_bead {
                CgPolicy::Single
            } else {
                CgPolicy::Multi
            },
            homo_dimer: s.homo_dimer,
        }
    }
}

impl Default for ScanInputs {
    fn default() -> Self {
        Self {
            rmin: 26.0,
            rmax: 60.0,
            dr: 1.0,
            molarity: 0.1,
            temperature: 298.15,
            max_ndiv: 2,
            subdivision: Subdivision::default(),
            gradient_threshold: 0.5,
            cutoff: 20.0,
            grid_size: 200,
            ph: 7.0,
            model: Model::Calvados3,
            cg: CgPolicy::Single,
            homo_dimer: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_from_archives_map_back_to_ui_models() {
        assert_eq!(Model::from_name("kimhummer"), Some(Model::KimHummer));
        assert_eq!(Model::from_name("Calvados3"), Some(Model::Calvados3));
        assert_eq!(Model::from_name("custom topology"), None);
    }

    #[test]
    fn imported_settings_restore_the_inputs_they_came_from() {
        let inputs = ScanInputs {
            model: Model::Pasquier,
            cg: CgPolicy::Multi,
            ph: 4.5,
            homo_dimer: true,
            subdivision: Subdivision::Lattice,
            ..Default::default()
        };
        let restored = ScanInputs::from_settings(&ScanSettings::from(&inputs), Model::Calvados3);
        assert_eq!(
            (restored.model, restored.cg, restored.ph),
            (Model::Pasquier, CgPolicy::Multi, 4.5)
        );
        assert_eq!(restored.subdivision, Subdivision::Lattice);
        assert!(restored.homo_dimer);
    }

    /// The panel must still show the scan's numbers when its model isn't selectable.
    #[test]
    fn unknown_models_keep_the_current_choice_but_restore_settings() {
        let mut settings = ScanSettings::from(&ScanInputs::default());
        settings.model = "custom topology".into();
        settings.molarity = 0.5;
        let restored = ScanInputs::from_settings(&settings, Model::KimHummer);
        assert_eq!(restored.model, Model::KimHummer);
        assert_eq!(restored.molarity, 0.5);
    }
}
