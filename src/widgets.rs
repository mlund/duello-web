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

/// Combo over `options`, shown by `label`, each with `tip` if given.
pub fn pick<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut T,
    options: &[T],
    label: impl Fn(&T) -> &str,
    tip: Option<fn(T) -> &'static str>,
) -> egui::Response {
    egui::ComboBox::from_id_salt(id)
        .selected_text(label(value))
        .show_ui(ui, |ui| {
            for option in options {
                let item = ui.selectable_value(value, *option, label(option));
                if let Some(tip) = tip {
                    item.on_hover_text(tip(*option));
                }
            }
        })
        .response
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

/// Refits `plot` to its data when `reset` is set, then clears it. Only then:
/// auto-bounds follows growing curves in between, so the user can still pan.
pub fn refit<'a>(plot: egui_plot::Plot<'a>, reset: &mut bool) -> egui_plot::Plot<'a> {
    if std::mem::take(reset) {
        plot.reset()
    } else {
        plot
    }
}

/// The tabs' main curve colour, so their plots read alike.
pub fn curve_color(dark_mode: bool) -> egui::Color32 {
    if dark_mode {
        egui::Color32::from_rgb(100, 150, 255)
    } else {
        egui::Color32::from_rgb(30, 60, 180)
    }
}

/// Centred hint where a window's content would be.
pub fn placeholder(ui: &mut egui::Ui, text: &str) {
    ui.centered_and_justified(|ui| ui.weak(text));
}

/// Gap and rule between panel sections.
pub fn section(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(2.0);
}
