//! A molecule as the 3D viewer draws it: beads or its surface, and its dipole.

use super::surface::{Field, Surface};
use crate::viewer::{categorical_color, Cylinder, Mesh, MeshVertex, Rgba, Scene, Sphere};
use duello::archive::CgBody;
use duello::molecule::Molecule;
use glam::Vec3;

/// What the bead colours show.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Colouring {
    Charge,
    Kind,
}

impl Colouring {
    pub const ALL: [Colouring; 2] = [Colouring::Charge, Colouring::Kind];

    pub fn label(self) -> &'static str {
        match self {
            Colouring::Charge => "Charge",
            Colouring::Kind => "Kind",
        }
    }

    pub fn tip(self) -> &'static str {
        match self {
            Colouring::Charge => "Red negative, grey neutral, blue positive; full at ±1 e",
            Colouring::Kind => "One colour per atom kind",
        }
    }
}

/// Beads, or the surface coloured by a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Show {
    Beads,
    Surface(Field),
}

impl Show {
    pub const ALL: [Show; 3] = [
        Show::Beads,
        Show::Surface(Field::Potential),
        Show::Surface(Field::Fluctuation),
    ];

    pub fn label(self) -> &'static str {
        match self {
            Show::Beads => "Beads",
            Show::Surface(Field::Potential) => "Surface φ",
            Show::Surface(Field::Fluctuation) => "Surface σ_φ",
        }
    }

    pub fn tip(self) -> &'static str {
        match self {
            Show::Beads => "One sphere per bead",
            Show::Surface(Field::Potential) => {
                "Mean screened (Debye–Hückel) potential φ one water radius outside \
                 the surface, kT/e: red negative, blue positive"
            }
            Show::Surface(Field::Fluctuation) => {
                "Spread σ_φ of that potential from titrating sites, kT/e: \
                 grey steady, gold to magenta fluctuating"
            }
        }
    }
}

/// Everything the viewer's choices change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Look {
    pub show: Show,
    /// Beads only.
    pub colouring: Colouring,
    pub dipole: bool,
}

impl Default for Look {
    fn default() -> Self {
        Self {
            show: Show::Beads,
            colouring: Colouring::Charge,
            dipole: false,
        }
    }
}

/// How far the dipole arrow reaches past the molecule, Å per e·Å: fixed, so
/// molecules compare. Measured past the molecule so even a weak one shows.
pub const ARROW_PER_DIPOLE: f64 = 0.2;

const ARROW_COLOR: Rgba = [0.25, 0.70, 0.35, 1.0];

/// Mass centre at the origin, as the body is built. Without a surface (it
/// failed) the beads stand in.
/// `mu` is the molecule's [`dipole`].
pub fn scene(molecule: &Molecule, mu: Vec3, look: Look, surface: Option<&Surface>) -> Scene {
    let mut scene = Scene::default();
    match (look.show, surface) {
        (Show::Surface(field), Some(surface)) => scene.mesh = surface_mesh(surface, field),
        _ => scene.spheres = beads(&molecule.body, look.colouring),
    }
    let radius = radius(&molecule.body);
    let tip = arrow_tip(mu, radius);
    if let Some(tip) = tip.filter(|_| look.dipole) {
        arrow(&mut scene, tip);
    }
    // The same for every look, so changing it keeps the user's camera.
    let reach = tip.map_or(radius, |t| t.length().max(radius));
    scene.bounds = ([0.0; 3], reach);
    scene
}

/// μ = Σ⟨qᵢ⟩rᵢ about the mass centre, e·Å: the mean charges' dipole, a
/// little under the ⟨|μ|⟩ that Multipole averages over states.
pub fn dipole(molecule: &Molecule) -> Vec3 {
    (molecule.body.positions.iter().zip(&molecule.beads))
        .map(|(r, bead)| Vec3::from(r.map(|x| x as f32)) * bead.charge as f32)
        .sum()
}

fn beads(body: &CgBody, colouring: Colouring) -> Vec<Sphere> {
    let colours: Vec<Rgba> = match colouring {
        Colouring::Charge => body.charges.iter().map(|&z| charge_color(z)).collect(),
        Colouring::Kind => kind_colours(&body.kinds),
    };
    (body.positions.iter().zip(&body.radii).zip(colours))
        .map(|((p, &r), color)| Sphere {
            center: p.map(|x| x as f32),
            radius: r as f32,
            color,
        })
        .collect()
}

fn surface_mesh(surface: &Surface, field: Field) -> Mesh {
    let mesh = surface.mesh();
    let rgba = surface.colours(field);
    let vertices = (mesh.positions.iter().zip(&mesh.normals).zip(rgba))
        .map(|((p, n), &color)| MeshVertex {
            position: p.map(|x| x as f32),
            normal: n.map(|x| x as f32),
            color,
        })
        .collect();
    Mesh {
        vertices,
        indices: mesh.triangles.iter().flatten().copied().collect(),
    }
}

/// Along μ, [`ARROW_PER_DIPOLE`] past `radius`; none without a dipole.
fn arrow_tip(mu: Vec3, radius: f32) -> Option<Vec3> {
    let axis = mu.try_normalize()?;
    Some(axis * (radius + mu.length() * ARROW_PER_DIPOLE as f32))
}

/// A shaft from the mass centre and a cone at `tip`.
fn arrow(scene: &mut Scene, tip: Vec3) {
    const SHAFT: f32 = 0.8;
    const HEAD: (f32, f32) = (5.0, 2.0);
    // Half the arrow at most, so the shaft never vanishes.
    let base = tip - tip.normalize() * HEAD.0.min(0.5 * tip.length());
    scene.cylinders.push(Cylinder {
        start: [0.0; 3],
        radius: SHAFT,
        end: base.into(),
        _pad: 0.0,
        start_color: ARROW_COLOR,
        end_color: ARROW_COLOR,
    });
    scene.mesh.append(&cone(base, tip, HEAD.1));
}

/// Side and base, smooth-shaded.
fn cone(base: Vec3, tip: Vec3, radius: f32) -> Mesh {
    const SEGMENTS: u32 = 24;
    let height = tip - base;
    let axis = height.normalize();
    let (u, v) = axis.any_orthonormal_pair();
    let mut mesh = Mesh::default();
    let mut push = |position: Vec3, normal: Vec3| {
        mesh.vertices.push(MeshVertex {
            position: position.into(),
            normal: normal.into(),
            color: ARROW_COLOR,
        });
    };
    push(base, -axis);
    for i in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
        let radial = u * angle.cos() + v * angle.sin();
        let rim = base + radial * radius;
        // Slanted normal of a cone of this height and radius.
        let normal = (radial * height.length() + axis * radius).normalize();
        push(rim, -axis);
        push(rim, normal);
        push(tip, normal);
    }
    // Per segment: base rim (cap), side rim, apex, in that order.
    let vertex = |i: u32, k: u32| 1 + 3 * (i % SEGMENTS) + k;
    for i in 0..SEGMENTS {
        mesh.indices.extend([0, vertex(i, 0), vertex(i + 1, 0)]);
        mesh.indices
            .extend([vertex(i, 1), vertex(i + 1, 1), vertex(i, 2)]);
    }
    mesh
}

/// Of the sphere about the mass centre that holds every bead, Å.
fn radius(body: &CgBody) -> f32 {
    let norm = |p: &[f64; 3]| p.iter().map(|x| x * x).sum::<f64>().sqrt();
    (body.positions.iter().zip(&body.radii))
        .map(|(p, r)| (norm(p) + r) as f32)
        .fold(1.0, f32::max)
}

/// One colour per kind, by first appearance.
fn kind_colours(kinds: &[String]) -> Vec<Rgba> {
    let mut seen: Vec<&str> = Vec::new();
    kinds
        .iter()
        .map(|kind| {
            let index = seen.iter().position(|k| k == kind).unwrap_or_else(|| {
                seen.push(kind);
                seen.len() - 1
            });
            categorical_color(index)
        })
        .collect()
}

/// Red for negative, blue for positive, saturating at one charge, so a bead's sign
/// reads at a glance; the surface potential uses the same map.
pub fn charge_color(z: f64) -> Rgba {
    cgkitten::coloring::diverging(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_colour_shows_the_sign_and_saturates_at_one_charge() {
        let [r, g, b, a] = charge_color(0.0);
        assert!(r == g && g == b && a == 1.0, "neutral is grey");
        let [r, _, b, _] = charge_color(-0.5);
        assert!(r > b, "negative is red");
        let [r, _, b, _] = charge_color(0.5);
        assert!(b > r, "positive is blue");
        assert_eq!(charge_color(3.0), charge_color(1.0));
        assert_eq!(charge_color(-3.0), charge_color(-1.0));
        assert_ne!(charge_color(0.5), charge_color(1.0));
    }
}
