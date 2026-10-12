//! Which release is running, and switching to another one.
//!
//! Each release is deployed, frozen, under `/duello-web/vX.Y.Z/` with a shared
//! `versions.json` one level up, so results can be reproduced with the exact build
//! that produced them.

use std::sync::mpsc;

/// This build, e.g. `v0.5.0`.
pub const APP: &str = concat!("v", env!("CARGO_PKG_VERSION"));
/// The duello library compiled in, read from `Cargo.lock` by `build.rs`.
pub const DUELLO: &str = env!("DUELLO_VERSION");

pub struct VersionPicker {
    /// Deployed releases, newest first; empty until fetched (or when not deployed).
    available: Vec<String>,
    rx: Option<mpsc::Receiver<Vec<String>>>,
    /// Chosen while results are shown; switching discards them, so ask first.
    pending: Option<String>,
}

impl VersionPicker {
    pub fn new() -> Self {
        Self {
            available: Vec::new(),
            rx: fetch_available(),
            pending: None,
        }
    }

    /// `has_results`: switching reloads the page, so confirm before discarding them.
    pub fn ui(&mut self, ui: &mut egui::Ui, has_results: bool) {
        if let Some(versions) = self.rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.available = versions;
            self.rx = None;
        }
        if self.rx.is_some() {
            // Nothing else wakes egui when the fetch completes.
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(250));
        }
        if self.available.is_empty() {
            ui.label(format!("Version {APP}"));
        } else {
            self.combo(ui, has_results);
        }
        ui.small(format!("duello {DUELLO}"));
        self.confirm(ui);
    }

    fn combo(&mut self, ui: &mut egui::Ui, has_results: bool) {
        let latest = self.available[0].clone();
        let label = |v: &str| {
            if v == latest {
                format!("{v} (latest)")
            } else {
                v.to_string()
            }
        };
        let mut chosen = None;
        ui.horizontal(|ui| {
            ui.label("Version");
            egui::ComboBox::from_id_salt("version_combo")
                .selected_text(label(APP))
                .show_ui(ui, |ui| {
                    for v in &self.available {
                        if ui.selectable_label(v == APP, label(v)).clicked() && v != APP {
                            chosen = Some(v.clone());
                        }
                    }
                });
        });
        match chosen {
            Some(v) if has_results => self.pending = Some(v),
            Some(v) => navigate_to(&v),
            None => {}
        }
    }

    fn confirm(&mut self, ui: &mut egui::Ui) {
        let Some(v) = self.pending.clone() else {
            return;
        };
        ui.label(format!("Switch to {v}? Current results will be lost."));
        ui.horizontal(|ui| {
            if ui.button("Switch").clicked() {
                navigate_to(&v);
            }
            if ui.button("Cancel").clicked() {
                self.pending = None;
            }
        });
    }
}

#[cfg(target_arch = "wasm32")]
fn fetch_available() -> Option<mpsc::Receiver<Vec<String>>> {
    let (tx, rx) = mpsc::channel();
    wasm_bindgen_futures::spawn_local(async move {
        // Missing outside a versioned deploy (e.g. `trunk serve`); the picker then
        // just shows this build's version.
        match crate::app::fetch_url("../versions.json").await {
            Ok(bytes) => match serde_json::from_slice::<Vec<String>>(&bytes) {
                Ok(versions) => {
                    let _ = tx.send(versions);
                }
                Err(e) => log::warn!("Bad versions.json: {e}"),
            },
            Err(e) => log::info!("No versions.json: {e}"),
        }
    });
    Some(rx)
}

/// Native builds have no deployed siblings to switch to.
#[cfg(not(target_arch = "wasm32"))]
fn fetch_available() -> Option<mpsc::Receiver<Vec<String>>> {
    None
}

#[cfg(target_arch = "wasm32")]
fn navigate_to(version: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.location().set_href(&format!("../{version}/"));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn navigate_to(_version: &str) {}
