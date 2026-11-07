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

/// Tooltip for a value shown read-only outside Onebody, which owns it.
pub fn set_in_onebody(tip: &str) -> String {
    format!("{tip}\n\nSet in Onebody.")
}

/// Lower right, clear of the low-q rise and g(r)'s contact peak; shared so the
/// tabs' plots read alike.
pub fn curve_legend() -> egui_plot::Legend {
    egui_plot::Legend::default().position(egui_plot::Corner::RightBottom)
}

/// Gap and rule between panel sections.
pub fn section(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(2.0);
}
