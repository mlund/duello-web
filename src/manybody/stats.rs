//! Run statistics and the downloads of what the run produced.

use super::RunInfo;
use crate::files::save_file;
use duello::nbody::{NBodyRun, Progress, Stage};
use std::borrow::Cow;

/// A download's outcome goes to `error`: set on failure, cleared on success.
pub fn show(
    ui: &mut egui::Ui,
    run: &NBodyRun,
    progress: Progress,
    info: RunInfo,
    error: &mut Option<String>,
) {
    summary(ui, progress, info);
    sampling(ui, run, progress, info);
    if !progress.recording {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Trajectory reached its size cap; recording stopped",
        );
    }
    match downloads(ui, run, progress) {
        Some(Err(e)) => *error = Some(format!("Save failed: {e:#}")),
        Some(Ok(())) => *error = None,
        None => {}
    }
}

fn stage_label(stage: Stage) -> &'static str {
    match stage {
        Stage::Equilibrating => "equilibrating",
        Stage::Production { equilibrated: true } => "production",
        Stage::Production {
            equilibrated: false,
        } => "production (not equilibrated: sweep cap reached)",
        Stage::Stopped => "stopped",
    }
}

fn summary(ui: &mut egui::Ui, progress: Progress, info: RunInfo) {
    ui.label(format!(
        "Run: N = {}, {:.2} mM protein, {:.0} mM ionic strength. Stage: {}; sweep {} per copy; {} frames; U/N = {:.2} kT",
        info.n_bodies,
        info.millimolar,
        info.ionic_millimolar,
        stage_label(progress.stage),
        progress.sweep,
        progress.frames,
        progress.energy_per_body
    ));
}

/// Move steps, acceptance and production energy statistics.
fn sampling(ui: &mut egui::Ui, run: &NBodyRun, progress: Progress, info: RunInfo) {
    let stats = run.stats();
    let n = info.n_bodies as f64;
    let show = |x: Option<f64>| x.map_or("–".to_string(), |v| format!("{v:.3}"));
    // Production statistics are empty until the first production frame.
    let produced = (stats.energy.frames > 0).then_some(());
    let (translation, rotation) = stats.steps;
    let tuning = if progress.stage == Stage::Equilibrating && info.tune_steps {
        " (tuning)"
    } else {
        ""
    };
    ui.label(format!(
        "Steps{tuning}: translate ±{translation:.2} Å, rotate ±{rotation:.3} rad"
    ));
    ui.label(format!(
        "Acceptance: translate {}, rotate {}; ⟨Δr²⟩ {} Å², ⟨Δθ²⟩ {} rad²",
        show(stats.moves.translate.acceptance()),
        show(stats.moves.rotate.acceptance()),
        show(stats.moves.translate.mean_square_displacement()),
        show(stats.moves.rotate.mean_square_displacement()),
    ));
    ui.label(format!(
        "⟨U⟩/N = {} kT, Var(U)/N = {} kT²",
        show(produced.map(|_| stats.energy.mean / n)),
        show(produced.map(|_| stats.energy.variance / n)),
    ));
}

#[derive(Clone, Copy)]
enum Download {
    Curve,
    PairCorrelation,
    Trajectory,
}

/// `Some` when a download was asked for: whether saving it worked.
fn downloads(ui: &mut egui::Ui, run: &NBodyRun, progress: Progress) -> Option<anyhow::Result<()>> {
    let mut pick = None;
    ui.horizontal(|ui| {
        let mut button = |enabled: bool, label: &str, hover: &str, kind: Download| {
            let clicked = ui
                .add_enabled(enabled, egui::Button::new(label))
                .on_hover_text(hover)
                .on_disabled_hover_text("Available once production frames exist")
                .clicked();
            if clicked {
                pick = Some(kind);
            }
        };
        button(
            true,
            "Curve (.csv)",
            "Scattering curves and run statistics",
            Download::Curve,
        );
        button(
            progress.frames > 0,
            "g(r) (.csv)",
            "Mass-centre radial distribution; error bars need two or more copies",
            Download::PairCorrelation,
        );
        button(
            true,
            "Trajectory (.traj)",
            "Production frames: positions and orientations",
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
