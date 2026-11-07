//! Which release is running, for the native app. The browser shows the same line in
//! the HTML header of `index.html`, which also switches between deployed releases and
//! keeps working even if an old build fails to start.

/// One line for the top bar, which lays out right to left.
pub fn ui(ui: &mut egui::Ui) {
    ui.hyperlink_to("GitHub", "https://github.com/mlund/duello-web");
    ui.weak(format!(
        "DuelloWeb v{} · duello {}",
        env!("CARGO_PKG_VERSION"),
        duello::VERSION
    ));
}
