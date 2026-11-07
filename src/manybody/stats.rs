//! Run statistics and the downloads of what the run produced.

use super::RunInfo;
use crate::files::save_file;
use crate::widgets::{grid, row};
use duello::nbody::{box_length, NBodyRun, Progress, RunStats, Stage};
use std::borrow::Cow;

/// A download's outcome goes to `error`: set on failure, cleared on success.
pub fn show(
    ui: &mut egui::Ui,
    run: &NBodyRun,
    progress: Progress,
    info: RunInfo,
    error: &mut Option<String>,
) {
    grid("mc_stats").striped(true).show(ui, |ui| {
        let stats = run.stats();
        run_rows(ui, progress);
        energy_rows(ui, &stats, progress, info);
        move_rows(ui, &stats, progress, info);
        system_rows(ui, info);
    });
    if info.record_trajectory && !progress.recording {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Trajectory reached its size cap; recording stopped",
        );
    }
    match downloads(ui, run, progress, info) {
        Some(Err(e)) => *error = Some(format!("Save failed: {e:#}")),
        Some(Ok(())) => *error = None,
        None => {}
    }
}

/// Read-only rows of (label, tooltip, value).
fn rows(ui: &mut egui::Ui, rows: &[(&str, &str, String)]) {
    for (label, tip, text) in rows {
        row(ui, label, tip, |ui| ui.label(text));
    }
}

fn show_or_dash(x: Option<f64>, digits: usize) -> String {
    x.map_or("–".to_string(), |v| format!("{v:.digits$}"))
}

pub(super) fn stage_label(stage: Stage) -> &'static str {
    match stage {
        Stage::Equilibrating => "equilibrating",
        Stage::Production { equilibrated: true } => "production",
        Stage::Production {
            equilibrated: false,
        } => "production (not equilibrated)",
        Stage::Stopped => "stopped",
    }
}

fn run_rows(ui: &mut egui::Ui, progress: Progress) {
    rows(
        ui,
        &[
            (
                "Stage",
                "Equilibration until the energy stops drifting, then production. \
                 \"Not equilibrated\": the sweep cap ended equilibration first.",
                stage_label(progress.stage).into(),
            ),
            (
                "Sweep",
                "Sweeps per copy so far; a sweep is one move per molecule",
                progress.sweep.to_string(),
            ),
            (
                "Frames",
                "Production frames sampled, summed over copies",
                progress.frames.to_string(),
            ),
        ],
    );
}

fn energy_rows(ui: &mut egui::Ui, stats: &RunStats, progress: Progress, info: RunInfo) {
    let n = info.n_bodies as f64;
    // Production statistics are empty until the first production frame.
    let produced = stats.energy.frames > 0;
    let per_body = |x: f64| show_or_dash(produced.then(|| x / n), 3);
    rows(
        ui,
        &[
            (
                "U/N",
                "Energy per molecule at the last check or frame, averaged over copies",
                format!("{:.3} kT", progress.energy_per_body),
            ),
            (
                "⟨U⟩/N",
                "Mean production energy per molecule",
                format!("{} kT", per_body(stats.energy.mean)),
            ),
            (
                "C_V,ex/(N k_B)",
                "Excess heat capacity per molecule from energy fluctuations, \
                 Var(U)/(N k_B²T²). It omits the ideal 3 k_B per rigid body and the \
                 temperature dependence of the effective potential (dielectric \
                 constant and screening), so it is an estimate.",
                per_body(stats.energy.variance),
            ),
        ],
    );
}

fn move_rows(ui: &mut egui::Ui, stats: &RunStats, progress: Progress, info: RunInfo) {
    let tuning = progress.stage == Stage::Equilibrating && info.tune_steps;
    let step = if tuning {
        "Largest move, still being tuned for the most displacement per attempt"
    } else {
        "Largest move, fixed for production"
    };
    let accepted = "Accepted fraction of attempted moves in production";
    let msd = "Mean square step per attempted move, rejected ones counting zero";
    let (t, r) = (&stats.moves.translate, &stats.moves.rotate);
    let (translation, rotation) = stats.steps;
    rows(
        ui,
        &[
            ("Translation step", step, format!("±{translation:.2} Å")),
            ("Rotation step", step, format!("±{rotation:.3} rad")),
            (
                "Translate acceptance",
                accepted,
                show_or_dash(t.acceptance(), 3),
            ),
            (
                "Rotate acceptance",
                accepted,
                show_or_dash(r.acceptance(), 3),
            ),
            (
                "⟨Δr²⟩",
                msd,
                format!("{} Å²", show_or_dash(t.mean_square_displacement(), 3)),
            ),
            (
                "⟨Δθ²⟩",
                msd,
                format!("{} rad²", show_or_dash(r.mean_square_displacement(), 4)),
            ),
        ],
    );
}

/// What the run was started with, which later edits don't change.
fn system_rows(ui: &mut egui::Ui, info: RunInfo) {
    let box_l = box_length(info.n_bodies, info.millimolar * 1e-3);
    rows(
        ui,
        &[
            (
                "Molecules, N",
                "Molecules per copy",
                info.n_bodies.to_string(),
            ),
            (
                "Copies",
                "Independent copies; their spread gives the error bars",
                info.copies.to_string(),
            ),
            (
                "Concentration",
                "Protein concentration of the run",
                format!("{:.2} mM", info.millimolar),
            ),
            (
                "Box L",
                "Side of the cubic periodic box",
                format!("{box_l:.0} Å"),
            ),
            (
                "Ionic strength",
                "Of the scan whose table the run uses",
                format!("{:.0} mM", info.ionic_millimolar),
            ),
        ],
    );
}

#[derive(Clone, Copy)]
enum Download {
    Curve,
    PairCorrelation,
    Trajectory,
}

/// `Some` when a download was asked for: whether saving it worked.
fn downloads(
    ui: &mut egui::Ui,
    run: &NBodyRun,
    progress: Progress,
    info: RunInfo,
) -> Option<anyhow::Result<()>> {
    const NO_FRAMES: &str = "Available once production frames exist";
    let mut pick = None;
    ui.horizontal(|ui| {
        let mut button = |enabled: bool, label: &str, hover: &str, why_not: &str, kind| {
            let clicked = ui
                .add_enabled(enabled, egui::Button::new(label))
                .on_hover_text(hover)
                .on_disabled_hover_text(why_not)
                .clicked();
            if clicked {
                pick = Some(kind);
            }
        };
        button(
            true,
            "Curve (.csv)",
            "Scattering curves and run statistics",
            NO_FRAMES,
            Download::Curve,
        );
        button(
            progress.frames > 0,
            "g(r) (.csv)",
            "Mass-centre radial distribution; error bars need two or more copies",
            NO_FRAMES,
            Download::PairCorrelation,
        );
        button(
            info.record_trajectory,
            "Trajectory (.traj)",
            "Production frames: positions and orientations",
            "Recording was off for this run: tick Record trajectory and run again",
            Download::Trajectory,
        );
    });
    Some(save(run, pick?))
}

fn save(run: &NBodyRun, kind: Download) -> anyhow::Result<()> {
    // Cow: the trajectory can be large, so it is saved without a copy.
    let (bytes, name): (Cow<[u8]>, _) = match kind {
        Download::Curve => (run.report_csv().into_bytes().into(), "manybody.csv"),
        Download::PairCorrelation => {
            let csv = run.pair_correlation().to_csv();
            (csv.into_bytes().into(), "manybody.rdf.csv")
        }
        Download::Trajectory => (run.trajectory_bytes()?, "manybody.traj"),
    };
    Ok(save_file(&bytes, name)?)
}
