//! Which release is running, for the native app. The browser shows the same line in
//! the HTML header of `index.html`, which also switches between deployed releases and
//! keeps working even if an old build fails to start.

/// This build, e.g. `v0.5.1`.
pub const APP: &str = concat!("v", env!("CARGO_PKG_VERSION"));
/// The duello library compiled in, read from `Cargo.lock` by `build.rs`.
pub const DUELLO: &str = env!("DUELLO_VERSION");

pub fn ui(ui: &mut egui::Ui) {
    ui.heading("DuelloWeb");
    ui.label(format!("{APP} · duello {DUELLO}"));
    ui.hyperlink_to("GitHub", "https://github.com/mlund/duello-web");
}
