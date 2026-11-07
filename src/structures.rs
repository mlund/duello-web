//! Structures loaded for the session, by RCSB fetch or file pick. Onebody
//! edits the list; other tabs select from it.

use duello::archive::StructureFile;
use std::sync::mpsc;

/// Survives removal of other entries, so a selection never slides onto a
/// different structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructureId(u64);

#[cfg(test)]
impl StructureId {
    pub fn for_tests(id: u64) -> Self {
        Self(id)
    }
}

/// A fetch or pick finishing off the UI thread.
struct Loaded {
    /// Only fetches count as pending: a cancelled pick never reports back.
    fetched: bool,
    /// `Err` explains a failure.
    file: Result<StructureFile, String>,
}

struct Entry {
    id: StructureId,
    /// `name` is what the user calls it, saved with a session.
    file: StructureFile,
    /// Restored when a rename is cleared: the name as loaded, or as saved in a
    /// session, which keeps no other.
    file_name: String,
}

pub struct Structures {
    entries: Vec<Entry>,
    next_id: u64,
    /// PDB ID typed for fetching.
    pdb_id: String,
    tx: mpsc::Sender<Loaded>,
    rx: mpsc::Receiver<Loaded>,
    /// Fetches in flight, shown so a slow one doesn't look ignored.
    pending: usize,
    error: Option<String>,
    /// May name a removed entry; [`Self::selected`] then falls back.
    selected: Option<StructureId>,
    /// Entry being renamed, with the text typed so far.
    editing: Option<(StructureId, String)>,
}

/// What a click in the list asks for, applied after the rows are drawn.
enum RowAction {
    Select(StructureId),
    Edit(StructureId),
    Rename(StructureId, String),
    CancelEdit,
    Remove(StructureId),
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
            selected: None,
            editing: None,
        }
    }
}

impl Structures {
    pub fn get(&self, id: StructureId) -> Option<&StructureFile> {
        self.entry(id).map(|e| &e.file)
    }

    fn entry(&self, id: StructureId) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    fn entry_mut(&mut self, id: StructureId) -> Option<&mut Entry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// Ids and names in load order.
    pub fn list(&self) -> impl Iterator<Item = (StructureId, &str)> {
        self.entries.iter().map(|e| (e.id, e.file.name.as_str()))
    }

    /// Position in load order, as a saved session indexes structures.
    pub fn index_of(&self, id: StructureId) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    /// Copies, in load order, for a saved session.
    pub fn files(&self) -> Vec<StructureFile> {
        self.entries.iter().map(|e| e.file.clone()).collect()
    }

    /// The entry the analysis shows: the one picked, else the first.
    pub fn selected(&self) -> Option<StructureId> {
        (self.selected)
            .filter(|&id| self.entry(id).is_some())
            .or_else(|| self.first())
    }

    pub fn select(&mut self, id: StructureId) {
        self.selected = Some(id);
    }

    pub fn remove(&mut self, id: StructureId) {
        self.entries.retain(|e| e.id != id);
    }

    /// A blank name falls back to the name it was loaded with.
    pub fn rename(&mut self, id: StructureId, name: &str) {
        if let Some(entry) = self.entry_mut(id) {
            let name = name.trim();
            entry.file.name = if name.is_empty() {
                entry.file_name.clone()
            } else {
                name.to_string()
            };
        }
    }

    /// Replaces the list with a saved session's, returning the new ids in
    /// the same order.
    pub fn replace(&mut self, files: Vec<StructureFile>) -> Vec<StructureId> {
        self.entries.clear();
        self.editing = None;
        files
            .into_iter()
            .map(|file| {
                self.push(file);
                self.entries.last().unwrap().id
            })
            .collect()
    }

    pub fn first(&self) -> Option<StructureId> {
        self.entries.first().map(|e| e.id)
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
        self.entries.push(Entry {
            id: StructureId(self.next_id),
            file_name: file.name.clone(),
            file,
        });
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
        let selected = self.selected();
        let mut action = None;
        for entry in &self.entries {
            let row = match &mut self.editing {
                Some((id, text)) if *id == entry.id => rename_row(ui, entry.id, text),
                _ => entry_row(ui, entry, selected == Some(entry.id)),
            };
            action = row.or(action);
        }
        match action {
            Some(RowAction::Select(id)) => self.select(id),
            Some(RowAction::Edit(id)) => {
                self.editing = self.get(id).map(|f| (id, f.name.clone()));
            }
            Some(RowAction::Rename(id, name)) => {
                self.rename(id, &name);
                self.editing = None;
            }
            Some(RowAction::CancelEdit) => self.editing = None,
            Some(RowAction::Remove(id)) => self.remove(id),
            None => {}
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

/// One entry: select by clicking its name, ✏ to rename, ✖ to remove.
fn entry_row(ui: &mut egui::Ui, entry: &Entry, selected: bool) -> Option<RowAction> {
    let file = &entry.file;
    ui.horizontal(|ui| {
        let name = ui
            .selectable_label(selected, &file.name)
            .on_hover_text(format!(
                "Loaded as {} ({} kB). Click to analyse it here; Twobody picks from this list.",
                entry.file_name,
                file.contents.len().div_ceil(1024)
            ));
        let edit = ui
            .small_button("✏")
            .on_hover_text("Rename; an empty name restores the file name");
        let remove = ui.small_button("✖").on_hover_text("Remove this structure");
        if remove.clicked() {
            Some(RowAction::Remove(entry.id))
        } else if edit.clicked() {
            Some(RowAction::Edit(entry.id))
        } else {
            name.clicked().then_some(RowAction::Select(entry.id))
        }
    })
    .inner
}

/// The entry being renamed, in place of its row. Enter or clicking away commits;
/// Escape cancels.
fn rename_row(ui: &mut egui::Ui, id: StructureId, text: &mut String) -> Option<RowAction> {
    let response = ui
        .add(egui::TextEdit::singleline(text).desired_width(160.0))
        .on_hover_text("Enter to rename, Escape to cancel; empty restores the file name");
    if !response.has_focus() && !response.lost_focus() {
        response.request_focus();
    }
    response.lost_focus().then(|| {
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            RowAction::CancelEdit
        } else {
            RowAction::Rename(id, text.clone())
        }
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, contents: &[u8]) -> StructureFile {
        StructureFile {
            name: name.into(),
            contents: contents.to_vec(),
        }
    }

    fn loaded(names: &[&str]) -> (Structures, Vec<StructureId>) {
        let mut structures = Structures::default();
        let ids = structures.replace(names.iter().map(|n| file(n, n.as_bytes())).collect());
        (structures, ids)
    }

    /// A saved session carries the custom name; clearing it brings the file name back.
    #[test]
    fn rename_keeps_id_and_contents_and_empty_restores_the_file_name() {
        let (mut structures, ids) = loaded(&["2CDS.pdb", "1UBQ.pdb"]);
        structures.rename(ids[0], "Lysozyme (pH 4)");
        assert_eq!(
            structures.get(ids[0]),
            Some(&file("Lysozyme (pH 4)", b"2CDS.pdb"))
        );
        assert_eq!(structures.files()[0].name, "Lysozyme (pH 4)");
        structures.rename(ids[0], "  ");
        assert_eq!(structures.get(ids[0]).unwrap().name, "2CDS.pdb");
        assert_eq!(structures.get(ids[1]).unwrap().name, "1UBQ.pdb");
    }

    /// The analysis follows a selection; removing it falls back to the first entry.
    #[test]
    fn selection_defaults_to_first_and_survives_removal_of_others() {
        let (mut structures, ids) = loaded(&["a.pdb", "b.pdb", "c.pdb"]);
        assert_eq!(structures.selected(), Some(ids[0]));
        structures.select(ids[2]);
        structures.remove(ids[1]);
        assert_eq!(structures.selected(), Some(ids[2]));
        structures.remove(ids[2]);
        assert_eq!(structures.selected(), Some(ids[0]));
        structures.remove(ids[0]);
        assert_eq!(structures.selected(), None);
    }
}
