//! Floating analysis windows over the Manybody canvas. One place sets them
//! up, so the five panes behave and tile alike.

use egui::{Rect, Ui};

/// Half the spacing between tiled panes, in points.
const GAP: f32 = 6.0;

#[derive(Clone, Copy, PartialEq, Debug)]
/// Declared in tiling order, as `ALL` lists them.
pub enum Pane {
    Simulation,
    Scattering,
    PairCorrelation,
    Energy,
    Stats,
}

impl Pane {
    pub const ALL: [Pane; 5] = [
        Pane::Simulation,
        Pane::Scattering,
        Pane::PairCorrelation,
        Pane::Energy,
        Pane::Stats,
    ];

    fn title(self) -> &'static str {
        match self {
            Pane::Energy => "Energy",
            Pane::Scattering => "Scattering",
            Pane::PairCorrelation => "g(r)",
            Pane::Stats => "MC stats",
            Pane::Simulation => "Simulation",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Pane::Energy => {
                "Energy per molecule, U/N: through equilibration, then afresh from the \
                 start of production"
            }
            Pane::Scattering => "Structure factor and, with Detailed I(q), I(q) and P(q)",
            Pane::PairCorrelation => "Radial distribution of mass centres, g(r)",
            Pane::Stats => "Run state, move acceptance, energy statistics and downloads",
            Pane::Simulation => "3D view of the simulation box",
        }
    }

    /// Fixed, not from the title, so retitling keeps users' layouts.
    fn id(self) -> egui::Id {
        egui::Id::unique(("manybody_pane", self as usize))
    }

    /// Where the pane first opens, tiled so none starts hidden under another.
    fn default_rect(self, area: Rect) -> Rect {
        // (column, row, columns spanned) on a 3 × 2 grid.
        let (col, row, span) = match self {
            Pane::Simulation => (0, 0, 1),
            Pane::Scattering => (1, 0, 1),
            Pane::PairCorrelation => (2, 0, 1),
            Pane::Energy => (0, 1, 1),
            Pane::Stats => (1, 1, 2),
        };
        let cell = area.size() / egui::vec2(3.0, 2.0);
        let min = area.min + cell * egui::vec2(col as f32, row as f32);
        let size = cell * egui::vec2(span as f32, 1.0);
        // A gap keeps neighbours (and their title bars) from touching.
        Rect::from_min_size(min, size).shrink(GAP)
    }
}

/// Shows `pane` as a window kept inside `ui`'s rect. `add_contents` runs
/// only while the window is expanded, so callers fetch their data inside
/// it and a collapsed pane costs no more than its title bar.
pub fn show(ui: &mut Ui, pane: Pane, add_contents: impl FnOnce(&mut Ui)) {
    let area = ui.max_rect();
    let shown = egui::Window::new(pane.title())
        .id(pane.id())
        .collapsible(true)
        .resizable(true)
        .constrain_to(area)
        .default_rect(pane.default_rect(area))
        .show(ui.ctx(), |ui| {
            let top = ui.max_rect().top();
            add_contents(ui);
            top
        });
    if let Some(shown) = shown {
        title_tooltip(ui, shown.response, shown.inner, pane);
    }
}

/// egui windows have no title tooltip; hovering the title bar stands in,
/// so the plots inside keep their own hover. The bar ends where the
/// contents start; a collapsed window is all bar.
fn title_tooltip(ui: &Ui, window: egui::Response, contents_top: Option<f32>, pane: Pane) {
    let mut title = window.rect;
    if let Some(top) = contents_top {
        title.max.y = top;
    }
    let over_title = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| title.contains(p));
    if over_title {
        window.on_hover_text(pane.tooltip());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> Rect {
        Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0))
    }

    #[test]
    fn every_pane_opens_inside_the_canvas() {
        for pane in Pane::ALL {
            let rect = pane.default_rect(area());
            assert!(area().contains_rect(rect), "{pane:?} at {rect:?}");
            assert!(
                rect.width() > 0.0 && rect.height() > 0.0,
                "{pane:?} is empty"
            );
        }
    }

    #[test]
    fn no_pane_opens_on_top_of_another() {
        for (i, a) in Pane::ALL.iter().enumerate() {
            for b in &Pane::ALL[i + 1..] {
                let (ra, rb) = (a.default_rect(area()), b.default_rect(area()));
                assert!(!ra.intersects(rb), "{a:?} {ra:?} overlaps {b:?} {rb:?}");
            }
        }
    }

    /// Frames in which `show` ran the pane's contents, with the pane's
    /// collapsed state stored up front as egui keeps it.
    fn content_runs(open: bool) -> usize {
        let ctx = egui::Context::default();
        // egui keys a window's collapsed state this way (window.rs, `show_dyn`).
        let collapsing = Pane::Scattering.id().with("collapsing");
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            &ctx, collapsing, true,
        );
        state.set_open(open);
        state.store(&ctx);
        let mut runs = 0;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                show(ui, Pane::Scattering, |_| runs += 1);
            });
        });
        // No renderer here to take the font atlas.
        output.textures_delta.clear();
        runs
    }

    #[test]
    fn a_collapsed_pane_does_no_work() {
        assert_eq!(content_runs(false), 0);
        assert_eq!(content_runs(true), 1);
    }
}
