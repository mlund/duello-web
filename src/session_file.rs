//! File ▸ Open/Save session: the dialogs, the browser/native split and the
//! zip, so the app only sees a [`Bundle`].

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use anyhow::Result;
use duello::bundle::Bundle;
use std::sync::mpsc;

const NAME: &str = "session.duello.zip";
/// `duello` too, so an old file gets the re-run message instead of being hidden.
const FILTER: (&str, &[&str]) = ("Duello session", &["zip", "duello"]);

/// A pick in progress. The browser reads files asynchronously, so both
/// targets hand the bytes over through a channel and [`Self::take_opened`].
#[derive(Default)]
pub struct SessionFile {
    picked: Option<mpsc::Receiver<Result<Vec<u8>>>>,
}

impl SessionFile {
    /// Native asks for the path before calling `zip`: zipping the table
    /// takes a while, and is wasted if the dialog is cancelled.
    pub fn save(zip: impl FnOnce() -> Result<Vec<u8>>) -> Result<()> {
        crate::files::save_with(NAME, zip)
    }

    pub fn open(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.picked = Some(rx);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(FILTER.0, FILTER.1)
            .pick_file()
        {
            let read = std::fs::read(&path).with_context(|| format!("reading {}", path.display()));
            let _ = tx.send(read);
        }
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let dialog = rfd::AsyncFileDialog::new().add_filter(FILTER.0, FILTER.1);
            if let Some(file) = dialog.pick_file().await {
                let _ = tx.send(Ok(file.read().await));
            }
        });
    }

    /// The picked session, once; a cancelled dialog yields nothing.
    pub fn take_opened(&mut self) -> Option<Result<Bundle>> {
        let received = self.picked.as_ref()?.try_recv();
        if !matches!(received, Err(mpsc::TryRecvError::Empty)) {
            self.picked = None;
        }
        Some(received.ok()?.and_then(|bytes| Bundle::from_zip(&bytes)))
    }
}
