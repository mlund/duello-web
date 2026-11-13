//! Floating analysis windows over a tab's canvas. One place sets them up, so
//! every tab's panes behave and tile alike.

use egui::{Rect, Ui};

/// Half the spacing between tiled panes, in points.
const GAP: f32 = 6.0;

/// One of a tab's windows.
pub trait Pane: Copy {
    /// Columns × rows the tab's panes tile.
    const GRID: [usize; 2];
    fn title(self) -> &'static str;
    /// Shown over the title bar.
    fn tooltip(self) -> &'static str;
    /// Where the pane first opens.
    fn slot(self) -> Slot;
}

/// A cell on the [`Pane::GRID`], `span` columns wide.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub col: usize,
    pub row: usize,
    pub span: usize,
}

/// From the type and the slot, not the title, so retitling keeps users' layouts.
fn id<P: Pane>(pane: P) -> egui::Id {
    let Slot { col, row, .. } = pane.slot();
    egui::Id::unique((std::any::type_name::<P>(), col, row))
}

/// Tiled so no pane starts hidden under another.
fn default_rect<P: Pane>(pane: P, area: Rect) -> Rect {
    let slot = pane.slot();
    let [columns, rows] = P::GRID;
    let cell = area.size() / egui::vec2(columns as f32, rows as f32);
    let min = area.min + cell * egui::vec2(slot.col as f32, slot.row as f32);
    let size = cell * egui::vec2(slot.span as f32, 1.0);
    // A gap keeps neighbours (and their title bars) from touching.
    Rect::from_min_size(min, size).shrink(GAP)
}

/// Shows `pane` as a window kept inside `ui`'s rect. `add_contents` runs
/// only while the window is expanded, so callers fetch their data inside
/// it and a collapsed pane costs no more than its title bar.
pub fn show<P: Pane>(ui: &mut Ui, pane: P, add_contents: impl FnOnce(&mut Ui)) {
    let area = ui.max_rect();
    let shown = egui::Window::new(pane.title())
        .id(id(pane))
        .collapsible(true)
        .resizable(true)
        .constrain_to(area)
        .default_rect(default_rect(pane, area))
        .show(ui.ctx(), |ui| {
            let top = ui.max_rect().top();
            add_contents(ui);
            top
        });
    if let Some(shown) = shown {
        title_tooltip(ui, shown.response, shown.inner, pane.tooltip());
    }
}

/// egui windows have no title tooltip; hovering the title bar stands in,
/// so the plots inside keep their own hover. The bar ends where the
/// contents start; a collapsed window is all bar.
fn title_tooltip(ui: &Ui, window: egui::Response, contents_top: Option<f32>, tip: &str) {
    let mut title = window.rect;
    if let Some(top) = contents_top {
        title.max.y = top;
    }
    let over_title = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| title.contains(p));
    if over_title {
        window.on_hover_text(tip);
    }
}

/// Every pane opens inside the canvas and on top of no other.
#[cfg(test)]
pub fn assert_tiled<P: Pane + std::fmt::Debug>(panes: &[P]) {
    let area = Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0));
    let rect = |pane: P| default_rect(pane, area);
    for (i, &a) in panes.iter().enumerate() {
        let ra = rect(a);
        assert!(area.contains_rect(ra), "{a:?} at {ra:?}");
        assert!(ra.width() > 0.0 && ra.height() > 0.0, "{a:?} is empty");
        for &b in &panes[i + 1..] {
            let rb = rect(b);
            assert!(!ra.intersects(rb), "{a:?} {ra:?} overlaps {b:?} {rb:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Only;

    impl Pane for Only {
        const GRID: [usize; 2] = [1, 1];
        fn title(self) -> &'static str {
            "Only"
        }
        fn tooltip(self) -> &'static str {
            ""
        }
        fn slot(self) -> Slot {
            Slot {
                col: 0,
                row: 0,
                span: 1,
            }
        }
    }

    /// Frames in which `show` ran the pane's contents, with the pane's
    /// collapsed state stored up front as egui keeps it.
    fn content_runs(open: bool) -> usize {
        let ctx = egui::Context::default();
        // egui keys a window's collapsed state this way (window.rs, `show_dyn`).
        let collapsing = id(Only).with("collapsing");
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            &ctx, collapsing, true,
        );
        state.set_open(open);
        state.store(&ctx);
        let mut runs = 0;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                show(ui, Only, |_| runs += 1);
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
