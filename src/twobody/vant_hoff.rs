//! Van 't Hoff fit of K_d over a temperature sweep, with a heat capacity term
//! so curvature in ln K_d against 1/T is fitted rather than averaged out.

use nalgebra::{DMatrix, DVector, Vector3};

/// ΔG°_d(T) = −RT ln K_d, K_d over a 1 M standard state, with ΔH°_d and ΔS°_d
/// drifting through a constant ΔC°_p,d about `t0`. Apparent: the model's
/// potential itself depends on T through ε_r and the Debye length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VantHoff {
    /// Reference temperature (K): the mean T fitted, where the values are least correlated.
    pub t0: f64,
    /// Dissociation enthalpy at `t0`, kJ/mol.
    pub dh_d: Estimate,
    /// Dissociation entropy at `t0`, J/(mol·K).
    pub ds_d: Estimate,
    /// Dissociation heat capacity, J/(mol·K).
    pub dcp_d: Estimate,
    /// Of ln K_d.
    pub r2: f64,
    /// Coefficients of `basis`: ΔH° in kJ/mol, ΔS° and ΔC°_p in kJ/(mol·K).
    coefficients: Vector3<f64>,
}

/// A fitted value and its standard error from the scatter about the fit;
/// over a narrow T range ΔC°_p is poorly determined, and this says so.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub value: f64,
    pub se: f64,
}

/// Three parameters need a fourth temperature before the scatter says anything.
const MIN_TEMPERATURES: usize = 4;

/// kJ to J for ΔS° and ΔC°_p.
const PER_KJ: f64 = 1e3;

/// Singular values below this fraction of the largest leave a direction unfitted.
const RANK_TOLERANCE: f64 = 1e-12;

impl VantHoff {
    /// Least squares of ln K_d over `[T (K), K_d (M)]`; points without a
    /// finite logarithm are skipped. The model is linear in ΔH°, ΔS° and
    /// ΔC°_p, so the solve is direct; SVD keeps the nearly collinear terms
    /// of a narrow T range from losing precision.
    pub fn fit(points: &[[f64; 2]]) -> Option<Self> {
        let usable: Vec<(f64, f64)> = points
            .iter()
            .map(|&[t, kd]| (t, kd.ln()))
            .filter(|(t, y)| *t > 0.0 && y.is_finite())
            .collect();
        if distinct_temperatures(&usable) < MIN_TEMPERATURES {
            return None;
        }
        let n = usable.len();
        let t0 = usable.iter().map(|p| p.0).sum::<f64>() / n as f64;
        let a = DMatrix::from_fn(n, 3, |i, j| basis(usable[i].0, t0)[j]);
        let y = DVector::from_iterator(n, usable.iter().map(|p| p.1));
        let svd = a.clone().svd(true, true);
        let s = &svd.singular_values;
        if s.min() <= RANK_TOLERANCE * s.max() {
            return None;
        }
        let x = svd.solve(&y, 0.0).ok()?;
        let rss = (&a * &x - &y).norm_squared();
        // Covariance σ²(AᵀA)⁻¹ = σ² V Σ⁻² Vᵀ.
        let v = svd.v_t?.transpose();
        let variance = rss / (n - 3) as f64;
        let estimate = |k: usize, scale: f64| {
            let diagonal: f64 = (0..3).map(|j| (v[(k, j)] / s[j]).powi(2)).sum();
            Estimate {
                value: x[k] * scale,
                se: (variance * diagonal).sqrt() * scale,
            }
        };
        Some(Self {
            t0,
            dh_d: estimate(0, 1.0),
            ds_d: estimate(1, PER_KJ),
            dcp_d: estimate(2, PER_KJ),
            r2: r_squared(&usable, rss),
            coefficients: Vector3::new(x[0], x[1], x[2]),
        })
    }

    /// Fitted ln K_d at `t` (K).
    pub fn line(&self, t: f64) -> f64 {
        Vector3::from(basis(t, self.t0)).dot(&self.coefficients)
    }

    /// One line for the pane.
    pub fn caption(&self) -> String {
        let [h, s, c] = [self.dh_d, self.ds_d, self.dcp_d];
        format!(
            "At {:.1} K: ΔH°_d = {:.1} ± {:.1} kJ/mol, ΔS°_d = {:.1} ± {:.1} J/(mol·K), \
             ΔC°_p,d = {:.0} ± {:.0} J/(mol·K), R² = {:.3}",
            self.t0, h.value, h.se, s.value, s.se, c.value, c.se, self.r2
        )
    }

    /// A trailing CSV comment, ASCII keys with units like the header.
    pub fn csv_comment(&self) -> String {
        let [h, s, c] = [self.dh_d, self.ds_d, self.dcp_d];
        format!(
            "# van 't Hoff: T0/K = {:.1}, dH_d/(kJ/mol) = {:.4} +- {:.4}, \
             dS_d/(J/(mol K)) = {:.4} +- {:.4}, dCp_d/(J/(mol K)) = {:.4} +- {:.4}, R2 = {:.6}\n",
            self.t0, h.value, h.se, s.value, s.se, c.value, c.se, self.r2
        )
    }
}

fn distinct_temperatures(points: &[(f64, f64)]) -> usize {
    let mut t: Vec<f64> = points.iter().map(|p| p.0).collect();
    t.sort_by(f64::total_cmp);
    t.dedup();
    t.len()
}

/// ln K_d = −ΔG°_d/RT per unit of (ΔH° kJ/mol, ΔS° and ΔC°_p kJ/(mol·K)), from
/// ΔH°(T) = ΔH°₀ + ΔC°_p(T − T₀) and ΔS°(T) = ΔS°₀ + ΔC°_p ln(T/T₀).
fn basis(t: f64, t0: f64) -> [f64; 3] {
    // R in kJ/(mol·K).
    let r = duello::thermal_energy(1.0);
    [-1.0 / (r * t), 1.0 / r, -((t - t0) / t - (t / t0).ln()) / r]
}

/// 1 when ln K_d doesn't vary: a flat line fits it exactly.
fn r_squared(points: &[(f64, f64)], rss: f64) -> f64 {
    let mean = points.iter().map(|p| p.1).sum::<f64>() / points.len() as f64;
    let total: f64 = points.iter().map(|(_, y)| (y - mean).powi(2)).sum();
    if total == 0.0 {
        1.0
    } else {
        1.0 - rss / total
    }
}

/// `[T, K_d]` rising with T at four temperatures; shared with the pane's tests.
#[cfg(test)]
pub(super) fn kd_by_t() -> Vec<[f64; 2]> {
    [280.0, 290.0, 300.0, 310.0]
        .iter()
        .zip([1e-6, 2e-6, 4e-6, 7e-6])
        .map(|(&t, k)| [t, k])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R in J/(mol·K), CODATA 2018.
    const R: f64 = 8.314462618;

    /// K_d from known values about T₀ = 300 K: ΔG°_d = ΔH° − TΔS° = −RT ln K_d.
    fn kd(t: f64, dh: f64, ds: f64, dcp: f64) -> f64 {
        let t0 = 300.0;
        let h = dh * 1e3 + dcp * (t - t0);
        let s = ds + dcp * (t / t0).ln();
        (-(h - t * s) / (R * t)).exp()
    }

    /// 280…320 K in 10 K steps, mean 300 K.
    fn sweep(dh: f64, ds: f64, dcp: f64) -> Vec<[f64; 2]> {
        (0..5)
            .map(|i| 280.0 + 10.0 * i as f64)
            .map(|t| [t, kd(t, dh, ds, dcp)])
            .collect()
    }

    #[test]
    fn exact_data_returns_enthalpy_entropy_and_heat_capacity() {
        let fit = VantHoff::fit(&sweep(24.1, 31.0, -1500.0)).expect("five points");
        assert_eq!(fit.t0, 300.0);
        assert!((fit.dh_d.value - 24.1).abs() < 1e-6, "{:?}", fit.dh_d);
        assert!((fit.ds_d.value - 31.0).abs() < 1e-6, "{:?}", fit.ds_d);
        assert!((fit.dcp_d.value + 1500.0).abs() < 1e-3, "{:?}", fit.dcp_d);
        assert!((fit.r2 - 1.0).abs() < 1e-12);
        let t = 287.0;
        assert!((fit.line(t) - kd(t, 24.1, 31.0, -1500.0).ln()).abs() < 1e-9);
    }

    /// Binding that weakens with T has K_d rising: ΔH°_d > 0. Guards the sign
    /// independently of `kd`, which shares the test's convention.
    #[test]
    fn k_d_rising_with_t_means_positive_dissociation_enthalpy() {
        assert!(VantHoff::fit(&kd_by_t()).unwrap().dh_d.value > 0.0);
    }

    /// Three points fit three parameters exactly, so R² would say nothing.
    #[test]
    fn fewer_than_four_points_give_no_fit() {
        assert!(VantHoff::fit(&sweep(24.1, 31.0, 0.0)[..3]).is_none());
    }

    /// K_d ≤ 0 or non-finite has no logarithm; the rest still fit.
    #[test]
    fn unusable_kd_is_skipped() {
        let mut points = sweep(24.1, 31.0, 0.0)[..4].to_vec();
        points.extend([[330.0, 0.0], [340.0, f64::INFINITY]]);
        let fit = VantHoff::fit(&points).expect("four usable points");
        assert_eq!(fit.t0, 295.0);
        assert!((fit.r2 - 1.0).abs() < 1e-9, "{fit:?}");
    }

    /// Exact data leaves no residual; scatter gives every value a spread.
    #[test]
    fn standard_errors_follow_the_scatter() {
        let exact = VantHoff::fit(&sweep(24.1, 31.0, -1500.0)).unwrap();
        assert!(exact.dh_d.se < 1e-6 && exact.dcp_d.se < 1e-3, "{exact:?}");
        let mut noisy = sweep(24.1, 31.0, -1500.0);
        noisy[1][1] *= 1.05;
        let fit = VantHoff::fit(&noisy).unwrap();
        for e in [fit.dh_d, fit.ds_d, fit.dcp_d] {
            assert!(e.se > 0.0 && e.se.is_finite(), "{fit:?}");
        }
    }

    /// Repeating a temperature adds no leverage on the T dependence.
    #[test]
    fn four_distinct_temperatures_are_needed() {
        let points = sweep(24.1, 31.0, 0.0);
        let repeated: Vec<[f64; 2]> = points[..3].iter().chain(&points[..1]).copied().collect();
        assert!(VantHoff::fit(&repeated).is_none());
    }

    /// K_d flat in T: the fit is exact, not undefined.
    #[test]
    fn flat_k_d_fits_with_r2_of_one() {
        let flat: Vec<[f64; 2]> = kd_by_t().iter().map(|&[t, _]| [t, 1e-6]).collect();
        let fit = VantHoff::fit(&flat).unwrap();
        assert_eq!(fit.r2, 1.0);
        assert!(fit.dh_d.value.abs() < 1e-6, "{fit:?}");
    }

    #[test]
    fn the_caption_states_the_fit_at_its_reference_temperature() {
        let estimate = |value, se| Estimate { value, se };
        let fit = VantHoff {
            t0: 298.15,
            dh_d: estimate(24.12, 0.31),
            ds_d: estimate(31.04, 1.2),
            dcp_d: estimate(-1520.0, 402.0),
            r2: 0.9981,
            coefficients: Vector3::zeros(),
        };
        assert_eq!(
            fit.caption(),
            "At 298.1 K: ΔH°_d = 24.1 ± 0.3 kJ/mol, ΔS°_d = 31.0 ± 1.2 J/(mol·K), \
             ΔC°_p,d = -1520 ± 402 J/(mol·K), R² = 0.998"
        );
        let line = fit.csv_comment();
        assert!(
            line.starts_with("# van 't Hoff: T0/K = 298.1, dH_d/(kJ/mol) = 24.1200 +- 0.3100, "),
            "{line}"
        );
        assert!(line.ends_with(", R2 = 0.998100\n"), "{line}");
    }
}
