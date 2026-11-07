use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use duello::web_api::ScanSettings;
use duello::Subdivision;

/// Twobody's own scan parameters; the solution and force field come from Onebody,
/// and a homo-dimer from picking one structure twice.
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
}

impl ScanInputs {
    /// `homo_dimer` is left false: only the structures can tell.
    pub fn settings(
        &self,
        conditions: &SolutionConditions,
        forcefield: &ForceField,
    ) -> ScanSettings {
        ScanSettings {
            ph: conditions.ph,
            model: forcefield.model.name().to_string(),
            single_bead: forcefield.single_bead(),
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
    /// [`SolutionConditions::adopt`] and [`ForceField::adopt`] take the rest.
    pub fn from_settings(s: &ScanSettings) -> Self {
        Self {
            rmin: s.rmin,
            rmax: s.rmax,
            dr: s.dr,
            max_ndiv: s.max_ndiv,
            subdivision: s.subdivision,
            gradient_threshold: s.gradient_threshold,
            cutoff: s.cutoff,
            grid_size: s.grid_size,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::{CgPolicy, Model};

    fn conditions() -> SolutionConditions {
        let mut conditions = SolutionConditions::default();
        conditions.temperature = 310.0;
        conditions.ph = 4.5;
        conditions.ionic_strength = 0.02;
        conditions
    }

    /// Onebody owns the solution and force field, so the scan takes them from there.
    #[test]
    fn settings_take_solution_and_force_field_from_onebody() {
        let forcefield = ForceField {
            model: Model::KimHummer,
            cg: CgPolicy::Multi,
        };
        let settings = ScanInputs::default().settings(&conditions(), &forcefield);
        assert_eq!(
            (settings.temperature, settings.ph, settings.ionic_strength),
            (310.0, 4.5, 0.02)
        );
        assert_eq!(
            (settings.model.as_str(), settings.single_bead),
            ("kimhummer", false)
        );
        assert_eq!(
            (settings.rmin, settings.rmax, settings.dr),
            (26.0, 60.0, 1.0)
        );
    }

    #[test]
    fn imported_settings_restore_the_inputs_and_conditions_they_came_from() {
        let inputs = ScanInputs {
            subdivision: Subdivision::Lattice,
            rmax: 80.0,
            ..Default::default()
        };
        let conditions = conditions();
        let settings = inputs.settings(&conditions, &ForceField::default());
        let restored = ScanInputs::from_settings(&settings);
        assert_eq!(
            (restored.subdivision, restored.rmax),
            (Subdivision::Lattice, 80.0)
        );
        let mut adopted = SolutionConditions::default();
        adopted.adopt(&settings);
        assert_eq!(adopted, conditions);
    }
}
