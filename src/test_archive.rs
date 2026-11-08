//! A small finished scan for tests, with a real table behind it.

use duello::archive::{CgBody, ScanArchive, ScanSettings};
use duello::web_api::PmfPoint;
use duello::{AdaptiveBuilder, VirialCoeff};

pub fn archive() -> ScanArchive {
    let settings = settings();
    let points: Vec<PmfPoint> = [(5.0, 1.0), (6.0, -0.5), (7.0, -0.1)]
        .map(|(r, w)| PmfPoint {
            r,
            free_energy: w,
            mean_energy: 2.0 * w,
            energy_variance: 0.5,
        })
        .into();
    let virial = VirialCoeff::from_pmf(points.iter().map(|p| (p.r, p.free_energy)), None)
        .expect("three points");
    ScanArchive {
        table: table(&settings),
        settings,
        bodies: [body(), body()],
        points,
        virial,
        topology_yaml: String::new(),
    }
}

fn settings() -> ScanSettings {
    ScanSettings {
        ph: 7.0,
        model: "test".into(),
        single_bead: true,
        rmin: 5.0,
        rmax: 8.0,
        dr: 1.0,
        ionic_strength: 0.1,
        temperature: 298.15,
        max_ndiv: 0,
        gradient_threshold: 0.5,
        cutoff: 30.0,
        grid_size: 100,
        homo_dimer: true,
        subdivision: Default::default(),
    }
}

/// Smooth energies, varying with both directions.
fn table(s: &ScanSettings) -> duello::Table6DAdaptive<f32> {
    let step = s.subdivision.angular_resolution(s.max_ndiv);
    let beta = 1.0 / duello::thermal_energy(s.temperature);
    let mut builder = AdaptiveBuilder::new(s.rmin, s.rmax, s.dr, step, s.max_ndiv, 1e-9, beta);
    for ri in 0..builder.n_r() {
        let verts = builder.vertex_directions(builder.current_level()).to_vec();
        for oi in 0..builder.n_omega() {
            let data: Vec<f64> = verts
                .iter()
                .flat_map(|a| verts.iter().map(move |b| 5.0 + a[0] - 2.0 * b[1]))
                .collect();
            builder.set_slab(ri, oi, &data);
        }
        builder.finish_r_slice(ri);
    }
    let mut table = builder.build();
    // Many-body runs read the scan temperature from here.
    let metadata = table.metadata.get_or_insert_with(Default::default);
    metadata.temperature = Some(s.temperature);
    metadata.point_group = match s.homo_dimer {
        true => duello::PointGroup::Exchange,
        false => duello::PointGroup::Asymmetric,
    };
    table
}

/// Mass centre at the origin, one bead of each sign.
fn body() -> CgBody {
    CgBody {
        positions: vec![[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        masses: vec![100.0, 100.0],
        charges: vec![1.0, -1.0],
        radii: vec![2.0, 2.0],
        kinds: vec!["LYS".into(), "GLU".into()],
    }
}
