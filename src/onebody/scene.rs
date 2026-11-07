//! A molecule as the 3D viewer draws it: one sphere per bead.

use crate::viewer::{categorical_color, Rgba, Scene, Sphere};
use duello::archive::CgBody;

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
}

/// Mass centre at the origin, as the body is built.
pub fn scene(body: &CgBody, colouring: Colouring) -> Scene {
    let colours: Vec<Rgba> = match colouring {
        Colouring::Charge => body.charges.iter().map(|&z| charge_color(z)).collect(),
        Colouring::Kind => kind_colours(&body.kinds),
    };
    let spheres: Vec<Sphere> = (body.positions.iter().zip(&body.radii).zip(colours))
        .map(|((p, &r), color)| Sphere {
            center: p.map(|x| x as f32),
            radius: r as f32,
            color,
        })
        .collect();
    let reach = spheres
        .iter()
        .map(|s| s.center.iter().map(|x| x * x).sum::<f32>().sqrt() + s.radius)
        .fold(1.0, f32::max);
    Scene {
        spheres,
        cylinders: Vec::new(),
        bounds: ([0.0; 3], reach),
    }
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
