//! The Manybody run's transport row: one main button that steps through
//! run, pause and resume, then stop and reset as icons. Always drawn, so the
//! panel below never shifts.

use egui::{pos2, vec2, Rect, Response, Ui};

/// Where the run is, as the buttons see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// No run, or a stopped one whose results stay.
    Idle,
    /// The GPU is being set up for a new run.
    Starting,
    Running,
    Paused,
}

/// What a click asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Start,
    Pause,
    Resume,
    /// Ends a run, or abandons its GPU setup.
    Stop,
    ResetStatistics,
}

/// Painted rather than typed: the app's font has no media glyphs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pictogram {
    Play,
    Pause,
    Stop,
}

/// Wide enough for its longest label, so it doesn't resize as it changes.
const MAIN_WIDTH: f32 = 110.0;

/// Room left of a label for its pictogram.
const PICTOGRAM_SPACE: &str = "      ";

/// The main button's pictogram, label and action; no action while the GPU
/// is set up.
fn primary(phase: Phase) -> (Pictogram, &'static str, Option<Action>) {
    match phase {
        Phase::Idle => (Pictogram::Play, "Run", Some(Action::Start)),
        Phase::Starting => (Pictogram::Play, "Starting…", None),
        Phase::Running => (Pictogram::Pause, "Pause", Some(Action::Pause)),
        Phase::Paused => (Pictogram::Play, "Resume", Some(Action::Resume)),
    }
}

fn primary_tip(phase: Phase) -> &'static str {
    match phase {
        Phase::Idle => "Equilibrate, then sample until stopped",
        Phase::Starting => "Preparing the bead sums on the GPU; the run starts when it is ready",
        Phase::Running => "Hold the run; resume where it left off",
        Phase::Paused => "Continue the held run",
    }
}

/// The row; `blocked` says why a run can't start, if it can't. At most one
/// action per frame.
pub fn ui(ui: &mut Ui, phase: Phase, blocked: Option<&str>, can_reset: bool) -> Option<Action> {
    let mut action = None;
    ui.horizontal(|ui| {
        let (pictogram, label, primary) = primary(phase);
        let enabled = primary.is_some() && !(phase == Phase::Idle && blocked.is_some());
        let text = format!("{PICTOGRAM_SPACE}{label}");
        let button = egui::Button::new(text).min_size(vec2(MAIN_WIDTH, 0.0));
        let response = ui
            .add_enabled(enabled, button)
            .on_hover_text(primary_tip(phase))
            .on_disabled_hover_text(blocked.unwrap_or(primary_tip(phase)));
        let left = pos2(
            response.rect.left() + 0.5 * response.rect.height(),
            response.rect.center().y,
        );
        paint(ui, &response, left, pictogram);
        if response.clicked() {
            action = primary;
        }
        let stop_tip = match phase {
            Phase::Starting => "Stop: abandon the GPU setup, e.g. if it never finishes",
            _ => "Stop: end the run; its results stay for saving",
        };
        let stop = square_button(ui, "", phase != Phase::Idle, stop_tip, "No run to stop");
        paint(ui, &stop, stop.rect.center(), Pictogram::Stop);
        if stop.clicked() {
            action = Some(Action::Stop);
        }
        let reset = square_button(
            ui,
            "↺",
            can_reset,
            "Reset statistics: discard production frames so far and keep sampling",
            "Reset statistics: available once production starts",
        );
        if reset.clicked() {
            action = Some(Action::ResetStatistics);
        }
    });
    action
}

fn square_button(ui: &mut Ui, text: &str, enabled: bool, tip: &str, off_tip: &str) -> Response {
    let button = egui::Button::new(egui::RichText::new(text).size(16.0)).min_size(vec2(28.0, 0.0));
    ui.add_enabled(enabled, button)
        .on_hover_text(tip)
        .on_disabled_hover_text(off_tip)
}

/// `pictogram` centred at `center` on `button`, in its text colour.
fn paint(ui: &Ui, button: &Response, center: egui::Pos2, pictogram: Pictogram) {
    let side = (button.rect.height() * 0.45).round();
    let colour = if button.enabled() {
        ui.style().interact(button).fg_stroke.color
    } else {
        ui.visuals().weak_text_color()
    };
    let at = |dx: f32, dy: f32| center + vec2(dx, dy) * side;
    let painter = ui.painter();
    match pictogram {
        Pictogram::Play => {
            let corners = vec![at(-0.4, -0.5), at(0.5, 0.0), at(-0.4, 0.5)];
            painter.add(egui::Shape::convex_polygon(
                corners,
                colour,
                egui::Stroke::NONE,
            ));
        }
        Pictogram::Pause => {
            for dx in [-0.25, 0.25] {
                let bar = Rect::from_center_size(at(dx, 0.0), vec2(0.22, 1.0) * side);
                painter.rect_filled(bar, 0.0, colour);
            }
        }
        Pictogram::Stop => {
            let square = Rect::from_center_size(center, vec2(0.85, 0.85) * side);
            painter.rect_filled(square, 1.0, colour);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_main_button_steps_through_run_pause_and_resume() {
        let shown = primary;
        assert_eq!(
            shown(Phase::Idle),
            (Pictogram::Play, "Run", Some(Action::Start))
        );
        assert_eq!(
            shown(Phase::Running),
            (Pictogram::Pause, "Pause", Some(Action::Pause))
        );
        assert_eq!(
            shown(Phase::Paused),
            (Pictogram::Play, "Resume", Some(Action::Resume))
        );
        assert_eq!(shown(Phase::Starting).2, None);
    }
}
