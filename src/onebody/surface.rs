//! The molecule's Gaussian surface, coloured by the screened potential its
//! titrated charges make there, or by that potential's fluctuation.

use crate::viewer::Rgba;
use cgkitten::potential::{PointCharge, SurfacePotential};
use cgkitten::surface::{gaussian_surface, Mesh};
use duello::archive::CgBody;
use duello::molecule::Molecule;
use std::rc::Rc;

/// What a surface colour shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    /// φ, kT/e.
    Potential,
    /// σ_φ, kT/e.
    Fluctuation,
}

/// Saturating at the 95th percentile lets most of a surface use the colour
/// range despite a few spikes; the floor keeps a near-neutral one pale.
const QUANTILE: f64 = 0.95;

impl Field {
    /// kT/e; σ_φ is an order smaller than φ, so it gets its own.
    fn floor(self) -> f64 {
        match self {
            Field::Potential => 0.5,
            Field::Fluctuation => 0.05,
        }
    }

    /// The colour scale in words, and how it is set.
    pub fn legend(self, scale: f64) -> (String, String) {
        let (text, of) = match self {
            Field::Potential => (format!("φ  red −{scale:.2} … blue +{scale:.2} kT/e"), "|φ|"),
            Field::Fluctuation => (format!("σ_φ  grey 0 … magenta {scale:.2} kT/e"), "σ_φ"),
        };
        let tip = format!(
            "Colours saturate at the {:.0}th percentile of {of}, at least {} kT/e",
            100.0 * QUANTILE,
            self.floor()
        );
        (text, tip)
    }
}

/// The magnitude of `values` that saturates `field`'s colours.
fn scale(values: &[f64], field: Field) -> f64 {
    let magnitudes = values.iter().map(|v| v.abs());
    cgkitten::coloring::percentile_scale(magnitudes, QUANTILE, field.floor())
}

/// Colours for `values` of `field`, and the value that saturates them.
pub fn colours(values: &[f64], field: Field) -> (Vec<Rgba>, f64) {
    use cgkitten::coloring::{diverging, sequential};
    let scale = scale(values, field);
    let map = match field {
        Field::Potential => diverging,
        Field::Fluctuation => sequential,
    };
    (values.iter().map(|v| map(v / scale)).collect(), scale)
}

/// One field over the surface, coloured once.
struct Painted {
    scale: f64,
    colours: Vec<Rgba>,
}

impl Painted {
    fn new(values: &[f64], field: Field) -> Self {
        let (colours, scale) = colours(values, field);
        Self { scale, colours }
    }
}

/// A molecule's surface and the field on it.
pub struct Surface {
    mesh: Rc<Mesh>,
    phi: Painted,
    sigma: Painted,
}

impl Surface {
    fn new(mesh: Rc<Mesh>, molecule: &Molecule, temperature: f64, ionic_strength: f64) -> Self {
        let potential = SurfacePotential::new(
            &point_charges(molecule),
            cgkitten::bjerrum_length(temperature),
            cgkitten::debye_length(temperature, ionic_strength),
        );
        let (phi, sigma): (Vec<_>, Vec<_>) = potential.on_mesh(&mesh).into_iter().unzip();
        Self {
            mesh,
            phi: Painted::new(&phi, Field::Potential),
            sigma: Painted::new(&sigma, Field::Fluctuation),
        }
    }

    fn painted(&self, field: Field) -> &Painted {
        match field {
            Field::Potential => &self.phi,
            Field::Fluctuation => &self.sigma,
        }
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// The value at which `field`'s colours saturate, kT/e.
    pub fn scale(&self, field: Field) -> f64 {
        self.painted(field).scale
    }

    /// One per vertex.
    pub fn colours(&self, field: Field) -> &[Rgba] {
        &self.painted(field).colours
    }
}

/// Bead rows, not the body: kinds merge the body's charges.
fn point_charges(molecule: &Molecule) -> Vec<PointCharge> {
    (molecule.body.positions.iter().zip(&molecule.beads))
        .map(|(&position, bead)| PointCharge {
            position,
            charge: bead.charge,
            variance: bead.charge_variance.unwrap_or(0.0),
        })
        .collect()
}

/// The mesh is slow (~80 ms natively for a 130-bead protein, once per
/// molecule) and the field a few ms; each is redone only when its inputs
/// change, so re-titrating keeps the mesh.
#[derive(Default)]
pub struct SurfaceView {
    /// Bead centres and radii, and their mesh.
    shape: Option<(Shape, Result<Rc<Mesh>, String>)>,
    /// By the generation of the molecule it was computed for.
    field: Option<(u64, Surface)>,
}

type Shape = (Vec<[f64; 3]>, Vec<f64>);

impl SurfaceView {
    /// `molecule`'s surface; `generation` names the molecule, titrated at
    /// `temperature` (K) and `ionic_strength` (mol/L), which also screen it.
    pub fn surface(
        &mut self,
        molecule: &Molecule,
        generation: u64,
        temperature: f64,
        ionic_strength: f64,
    ) -> Result<&Surface, String> {
        if self.field.as_ref().is_none_or(|(g, _)| *g != generation) {
            let mesh = self.mesh(&molecule.body)?;
            let surface = Surface::new(mesh, molecule, temperature, ionic_strength);
            self.field = Some((generation, surface));
        }
        Ok(&self.field.as_ref().expect("set above").1)
    }

    /// Kept across generations while the beads don't move.
    fn mesh(&mut self, body: &CgBody) -> Result<Rc<Mesh>, String> {
        let same = |(s, _): &(Shape, _)| s.0 == body.positions && s.1 == body.radii;
        if !self.shape.as_ref().is_some_and(same) {
            let mesh = gaussian_surface(&body.positions, &body.radii).map(Rc::new);
            self.shape = Some(((body.positions.clone(), body.radii.clone()), mesh));
        }
        self.shape.as_ref().expect("set above").1.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgkitten::coloring::{diverging, sequential};
    use duello::molecule::{MoleculeSpec, Step, Titrating};

    /// The tripeptide, Henderson–Hasselbalch-titrated at pH 7.
    fn molecule(ionic_strength: f64) -> Molecule {
        let spec = MoleculeSpec {
            model: "calvados3".into(),
            single_bead: true,
            ph: 7.0,
            ionic_strength,
            temperature: 298.15,
            mc_sweeps: 0,
        };
        let pdb = crate::onebody::tests::tripeptide();
        match Titrating::start(&pdb, spec, 1).unwrap().advance(usize::MAX) {
            Ok(Step::Done(molecule)) => *molecule,
            _ => panic!("HH titrates at once"),
        }
    }

    /// Salt changes the charges and the screening, not the shape.
    #[test]
    fn retitrating_under_more_salt_keeps_the_mesh_and_shrinks_the_potential_scale() {
        let mut view = SurfaceView::default();
        let low = view.surface(&molecule(0.005), 1, 298.15, 0.005).unwrap();
        let (mesh, low) = (low.mesh() as *const Mesh, low.scale(Field::Potential));
        let high = view.surface(&molecule(0.5), 2, 298.15, 0.5).unwrap();
        assert!(std::ptr::eq(mesh, high.mesh()), "mesh rebuilt");
        assert!(!high.mesh().triangles.is_empty());
        let high = high.scale(Field::Potential);
        // Above the floor, else the scale wouldn't show screening at all.
        assert!(high > Field::Potential.floor(), "{high}");
        assert!(high < low, "{low} → {high}");
    }

    /// 0, ±1, …, ±100: |φ| at the 95th percentile of 201 values is 95.
    #[test]
    fn potential_saturates_at_the_95th_percentile_of_its_magnitude() {
        let phi: Vec<f64> = (-100..=100).map(f64::from).collect();
        let (rgba, scale) = colours(&phi, Field::Potential);
        assert_eq!(scale, 95.0);
        assert_eq!(rgba[100], diverging(0.0), "φ = 0 is grey");
        assert_eq!(rgba[200], diverging(1.0), "φ = 100 saturates blue");
        assert_eq!(rgba[0], diverging(-1.0), "φ = −100 saturates red");
        assert_eq!(rgba[195], diverging(1.0), "φ = 95 just saturates");
    }

    /// A nearly neutral surface stays pale rather than stretching noise to full colour.
    #[test]
    fn small_values_are_scaled_against_a_floor() {
        let (rgba, scale) = colours(&[0.01, -0.01], Field::Potential);
        assert_eq!(scale, 0.5);
        assert_eq!(rgba[0], diverging(0.02));
        let (rgba, scale) = colours(&[0.01], Field::Fluctuation);
        assert_eq!(scale, 0.05);
        assert_eq!(rgba[0], sequential(0.2));
    }
}
