//! Structures loaded for the session, by RCSB fetch or file pick. Onebody
//! edits the list; other tabs select from it.

use duello::archive::StructureFile;
use std::sync::mpsc;

/// Survives removal of other entries, so a selection never slides onto a
/// different structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructureId(u64);

/// A fetch or pick finishing off the UI thread.
struct Loaded {
    /// Only fetches count as pending: a cancelled pick never reports back.
    fetched: bool,
    /// `Err` explains a failure.
    file: Result<StructureFile, String>,
}

pub struct Structures {
    entries: Vec<(StructureId, StructureFile)>,
    next_id: u64,
    /// PDB ID typed for fetching.
    pdb_id: String,
    tx: mpsc::Sender<Loaded>,
    rx: mpsc::Receiver<Loaded>,
    /// Fetches in flight, shown so a slow one doesn't look ignored.
    pending: usize,
    error: Option<String>,
}

impl Default for Structures {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            entries: Vec::new(),
            next_id: 0,
            pdb_id: "2CDS".into(),
            tx,
            rx,
            pending: 0,
            error: None,
        }
    }
}

impl Structures {
    pub fn get(&self, id: StructureId) -> Option<&StructureFile> {
        self.entries.iter().find(|(i, _)| *i == id).map(|(_, f)| f)
    }

    /// Ids and names in load order.
    pub fn list(&self) -> impl Iterator<Item = (StructureId, &str)> {
        self.entries.iter().map(|(id, f)| (*id, f.name.as_str()))
    }

    pub fn first(&self) -> Option<StructureId> {
        self.entries.first().map(|(id, _)| *id)
    }

    /// Takes finished fetches and picks; call every frame.
    pub fn poll(&mut self) {
        while let Ok(loaded) = self.rx.try_recv() {
            self.pending -= usize::from(loaded.fetched);
            match loaded.file {
                Ok(file) => self.push(file),
                Err(e) => self.error = Some(e),
            }
        }
    }

    fn push(&mut self, file: StructureFile) {
        self.entries.push((StructureId(self.next_id), file));
        self.next_id += 1;
        self.error = None;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.list_ui(ui);
        ui.add_space(4.0);
        self.add_ui(ui);
        if self.pending > 0 {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Loading…");
            });
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, e);
        }
    }

    fn list_ui(&mut self, ui: &mut egui::Ui) {
        if self.entries.is_empty() {
            ui.weak("None loaded")
                .on_hover_text("Add a structure below; Twobody scans one or two of them");
        }
        let mut removed = None;
        for (id, file) in &self.entries {
            ui.horizontal(|ui| {
                ui.label(&file.name).on_hover_text(format!(
                    "{} kB as loaded; select it as a molecule in Twobody",
                    file.contents.len().div_ceil(1024)
                ));
                if ui
                    .small_button("✖")
                    .on_hover_text("Remove this structure")
                    .clicked()
                {
                    removed = Some(*id);
                }
            });
        }
        if let Some(id) = removed {
            self.entries.retain(|(i, _)| *i != id);
        }
    }

    fn add_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button("Upload file...")
                .on_hover_text("Load a PDB file from disk")
                .clicked()
            {
                self.pick_file();
            }
            ui.label("or");
            ui.add(
                egui::TextEdit::singleline(&mut self.pdb_id)
                    .hint_text("PDB ID")
                    .desired_width(50.0),
            )
            .on_hover_text("Four-character RCSB PDB code, e.g. 4LZT");
            let id = self.pdb_id.trim().to_uppercase();
            let id_valid = id.len() == 4 && id.chars().all(|c| c.is_alphanumeric());
            if ui
                .add_enabled(id_valid, egui::Button::new("Fetch"))
                .on_hover_text("Download this structure from RCSB")
                .on_disabled_hover_text("Enter a four-character PDB ID")
                .clicked()
            {
                self.fetch_pdb(id);
            }
        });
    }

    fn fetch_pdb(&mut self, pdb_id: String) {
        self.pending += 1;
        let tx = self.tx.clone();
        let url = format!("https://files.rcsb.org/download/{pdb_id}.pdb");
        let named = move |data: anyhow::Result<Vec<u8>>| Loaded {
            fetched: true,
            file: data
                .map(|contents| StructureFile {
                    name: format!("{pdb_id}.pdb"),
                    contents,
                })
                .map_err(|e| format!("Fetching {pdb_id} failed: {e:#}")),
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let _ = tx.send(named(fetch_url(&url).await));
        });
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let _ = tx.send(named(fetch_url(&url)));
        });
    }

    /// Native: a blocking dialog, so the file lands at once.
    #[cfg(not(target_arch = "wasm32"))]
    fn pick_file(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PDB files", &["pdb"])
            .pick_file()
        else {
            return;
        };
        match std::fs::read(&path) {
            Ok(contents) => self.push(StructureFile {
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                contents,
            }),
            Err(e) => self.error = Some(format!("Reading {} failed: {e}", path.display())),
        }
    }

    /// Browser: the file is read asynchronously; a cancelled dialog sends nothing.
    #[cfg(target_arch = "wasm32")]
    fn pick_file(&mut self) {
        let tx = self.tx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let dialog = rfd::AsyncFileDialog::new().add_filter("PDB files", &["pdb"]);
            if let Some(file) = dialog.pick_file().await {
                let contents = file.read().await;
                let _ = tx.send(Loaded {
                    fetched: false,
                    file: Ok(StructureFile {
                        name: file.file_name(),
                        contents,
                    }),
                });
            }
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_url(url: &str) -> anyhow::Result<Vec<u8>> {
    let response = ureq::get(url).call()?;
    let mut data = Vec::new();
    std::io::Read::read_to_end(&mut response.into_body().as_reader(), &mut data)?;
    Ok(data)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_url(url: &str) -> anyhow::Result<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| anyhow::anyhow!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("not a Response"))?;
    if !resp.ok() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    let buf = JsFuture::from(
        resp.array_buffer()
            .map_err(|_| anyhow::anyhow!("no array_buffer"))?,
    )
    .await
    .map_err(|e| anyhow::anyhow!("array_buffer failed: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}
