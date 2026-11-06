//! A copy's configuration as something the 3D viewer can draw: beads as
//! spheres, coloured by molecule, inside the box outline.

use crate::viewer::{Cylinder, Rgba, Scene, Sphere};
use duello::archive::CgBody;
use duello::nbody::trajectory::expand;
use duello::nbody::Frame;

pub fn scene(frame: &Frame, body: &CgBody) -> Scene {
    let l = frame.box_length as f32;
    Scene {
        spheres: beads(frame, body),
        cylinders: box_edges(l),
        bounds: ([0.5 * l; 3], 3f32.sqrt() * 0.5 * l),
    }
}

/// One sphere per bead, coloured by molecule.
fn beads(frame: &Frame, body: &CgBody) -> Vec<Sphere> {
    let mut spheres = Vec::with_capacity(frame.poses.len() * body.radii.len());
    for (molecule, pose) in frame.poses.iter().enumerate() {
        let color = molecule_color(molecule);
        let placed = expand(body, pose).into_iter().zip(&body.radii);
        spheres.extend(placed.map(|(bead, &radius)| Sphere {
            center: bead.map(|x| x as f32).into(),
            radius: radius as f32,
            color,
        }));
    }
    spheres
}

/// The 12 edges of [0, l]³: corners `i` and `i | 1 << axis` differ along
/// one axis only. Thickness scales with the box so it reads at any size.
fn box_edges(l: f32) -> Vec<Cylinder> {
    let corner = |i: usize| [0, 1, 2].map(|axis| if i >> axis & 1 == 1 { l } else { 0.0 });
    const GREY: Rgba = [0.6, 0.6, 0.6, 1.0];
    (0..8)
        .flat_map(|i| {
            (0..3)
                .filter(move |axis| i >> axis & 1 == 0)
                .map(move |axis| (i, i | 1 << axis))
        })
        .map(|(a, b)| Cylinder {
            start: corner(a),
            radius: l / 500.0,
            end: corner(b),
            _pad: 0.0,
            start_color: GREY,
            end_color: GREY,
        })
        .collect()
}

/// Golden-angle hue steps keep any two nearby indices far apart in colour,
/// however many molecules there are.
fn molecule_color(index: usize) -> Rgba {
    const GOLDEN_TURN: f32 = 0.381_966;
    let hue = (index as f32 * GOLDEN_TURN).fract();
    // The shaders write display values, so convert out of egui's linear space.
    egui::ecolor::Hsva::new(hue, 0.6, 0.9, 1.0)
        .to_srgba_unmultiplied()
        .map(|c| f32::from(c) / 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use duello::nbody::Pose;
    use duello::{UnitQuaternion, Vector3};

    /// Three beads of radii 1, 2, 3 Å along x.
    fn body() -> CgBody {
        CgBody {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            masses: vec![1.0; 3],
            charges: vec![0.0; 3],
            radii: vec![1.0, 2.0, 3.0],
            kinds: vec!["X".into(); 3],
        }
    }

    fn at(position: [f64; 3], orientation: UnitQuaternion) -> Pose {
        Pose {
            position: Vector3::from(position),
            orientation,
        }
    }

    fn two_molecules() -> Frame {
        Frame {
            copy: 0,
            sweep: 0,
            box_length: 50.0,
            poses: vec![
                at([10.0, 10.0, 10.0], UnitQuaternion::identity()),
                at([30.0, 30.0, 30.0], UnitQuaternion::identity()),
            ],
        }
    }

    #[test]
    fn every_bead_is_a_sphere_of_its_radius() {
        let spheres = scene(&two_molecules(), &body()).spheres;
        let radii: Vec<f32> = spheres.iter().map(|s| s.radius).collect();
        assert_eq!(radii, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn each_molecule_has_its_own_colour() {
        let spheres = scene(&two_molecules(), &body()).spheres;
        let colours: Vec<_> = spheres.iter().map(|s| s.color).collect();
        assert!(colours[..3].iter().all(|c| *c == colours[0]));
        assert!(colours[3..].iter().all(|c| *c == colours[3]));
        assert_ne!(colours[0], colours[3]);
    }

    /// Beads follow the pose: (1, 0, 0) on a molecule turned 90° about z
    /// and placed at (5, 5, 5) is drawn at (5, 6, 5).
    #[test]
    fn beads_are_drawn_where_the_pose_puts_them() {
        let turn = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let frame = Frame {
            poses: vec![at([5.0, 5.0, 5.0], turn)],
            ..two_molecules()
        };
        let centre = scene(&frame, &body()).spheres[1].center;
        let expected = [5.0, 6.0, 5.0];
        assert!(
            centre
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-5),
            "{centre:?}"
        );
    }

    /// The box [0, L]³ with L = 50: 12 edges of length 50 between its 8
    /// corners, framed by the sphere through the corners, radius √3·25.
    #[test]
    fn the_box_is_outlined_and_framed() {
        let drawn = scene(&two_molecules(), &body());
        assert_eq!(drawn.cylinders.len(), 12);
        let mut corners: Vec<[f32; 3]> = Vec::new();
        for c in &drawn.cylinders {
            let length = (0..3)
                .map(|k| (c.end[k] - c.start[k]).powi(2))
                .sum::<f32>()
                .sqrt();
            assert_eq!(length, 50.0);
            for p in [c.start, c.end] {
                assert!(p.iter().all(|&x| x == 0.0 || x == 50.0), "{p:?}");
                if !corners.contains(&p) {
                    corners.push(p);
                }
            }
        }
        assert_eq!(corners.len(), 8);
        let (centre, radius) = drawn.bounds;
        assert_eq!(centre, [25.0; 3]);
        assert!((radius - 3f32.sqrt() * 25.0).abs() < 1e-4);
    }
}
