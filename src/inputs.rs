use crate::conditions::SolutionConditions;
use crate::forcefield::ForceField;
use duello::web_api::{GridSpec, ScanBackend, ScanSettings};
use duello::Subdivision;

/// Twobody's own scan parameters; the solution and force field come from Onebody,
/// and a homo-dimer from picking one structure twice.
#[derive(Clone)]
pub struct ScanInputs {
    pub rmin: f64,
    /// Let the scan choose `rmin` from bead overlap; the scan writes back what it used.
    pub rmin_auto: bool,
    pub rmax: f64,
    pub dr: f64,
    pub max_ndiv: usize,
    pub subdivision: Subdivision,
    pub gradient_threshold: f64,
    pub cutoff: f64,
    pub grid_size: usize,
    /// Scan on precomputed short-range maps (fast, slightly biased near
    /// contact) rather than summing exact pairs on the GPU.
    pub use_grid: bool,
    /// Kept while exact pairs are chosen, so switching back restores it.
    pub grid: GridSpec,
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
            rmin_auto: self.rmin_auto,
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

    /// The backend to request: `None` sums exact pairs.
    pub fn grid_choice(&self) -> Option<GridSpec> {
        self.use_grid.then_some(self.grid)
    }

    /// Match the backend an opened scan was made with. Sessions save only the
    /// settings, so this comes from the scan itself.
    pub fn adopt_backend(&mut self, backend: ScanBackend) {
        match backend {
            ScanBackend::Grid(spec) => (self.use_grid, self.grid) = (true, spec),
            ScanBackend::Gpu | ScanBackend::Simd | ScanBackend::Reference => self.use_grid = false,
        }
    }

    /// Match an imported scan's settings, so the panel describes what's shown;
    /// [`SolutionConditions::adopt`] and [`ForceField::adopt`] take the rest.
    /// The backend choice is kept: settings don't record it.
    pub fn adopt_settings(&mut self, s: &ScanSettings) {
        self.rmin = s.rmin;
        self.rmin_auto = s.rmin_auto;
        self.rmax = s.rmax;
        self.dr = s.dr;
        self.max_ndiv = s.max_ndiv;
        self.subdivision = s.subdivision;
        self.gradient_threshold = s.gradient_threshold;
        self.cutoff = s.cutoff;
        self.grid_size = s.grid_size;
    }
}

impl Default for ScanInputs {
    fn default() -> Self {
        Self {
            rmin: 26.0,
            rmin_auto: true,
            rmax: 60.0,
            dr: 1.0,
            max_ndiv: 2,
            subdivision: Subdivision::default(),
            gradient_threshold: 0.5,
            cutoff: 20.0,
            grid_size: 200,
            use_grid: true,
            grid: GridSpec::default(),
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

    /// The web app scans like `duello scan`: the grid at the CLI's spacing.
    #[test]
    fn default_inputs_use_the_cli_grid() {
        assert_eq!(
            ScanInputs::default().grid_choice(),
            Some(GridSpec::default())
        );
    }

    /// An opened scan shows the backend that made it; a later switch back to
    /// the grid keeps its spacing.
    #[test]
    fn opened_scan_sets_the_grid_choice() {
        let spec = GridSpec {
            spacing: 0.5,
            ..Default::default()
        };
        let mut inputs = ScanInputs::default();
        inputs.adopt_backend(ScanBackend::Grid(spec));
        assert_eq!(inputs.grid_choice(), Some(spec));
        for exact in [ScanBackend::Gpu, ScanBackend::Simd, ScanBackend::Reference] {
            inputs.adopt_backend(exact);
            assert_eq!(inputs.grid_choice(), None);
        }
        inputs.use_grid = true;
        assert_eq!(inputs.grid_choice(), Some(spec));
    }

    /// Sessions save settings, not the backend, so adopting them must not
    /// reset the user's choice.
    #[test]
    fn adopting_settings_keeps_the_backend_choice() {
        let mut inputs = ScanInputs {
            use_grid: false,
            ..Default::default()
        };
        let settings = ScanInputs::default().settings(&conditions(), &ForceField::default());
        inputs.adopt_settings(&settings);
        assert_eq!(inputs.grid_choice(), None);
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
        let mut restored = ScanInputs::default();
        restored.adopt_settings(&settings);
        assert_eq!(
            (restored.subdivision, restored.rmax),
            (Subdivision::Lattice, 80.0)
        );
        let mut adopted = SolutionConditions::default();
        adopted.adopt(&settings);
        assert_eq!(adopted, conditions);
    }
}
