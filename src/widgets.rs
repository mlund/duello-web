//! Panel building blocks shared by the tabs, so their layout stays alike.

/// Two-column label/value grid with the panels' spacing.
pub fn grid(id: &str) -> egui::Grid {
    egui::Grid::new(id).num_columns(2).spacing([8.0, 4.0])
}

/// A grid row whose label and widget share the tooltip, so hovering either
/// explains the value.
pub fn row(
    ui: &mut egui::Ui,
    label: &str,
    tip: &str,
    widget: impl FnOnce(&mut egui::Ui) -> egui::Response,
) {
    ui.label(label).on_hover_text(tip);
    widget(ui).on_hover_text(tip);
    ui.end_row();
}
