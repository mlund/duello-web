use duello::archive::StructureFile;
use duello::web_api::{GridSpec, ScanProgress, ScanSettings, WebScanRequest, WebScanResult};
use std::ops::ControlFlow;

/// Status of the computation.
pub enum ComputeStatus {
    Idle,
    Running { current_r: usize, total_r: usize },
}

pub type ComputeResult = WebScanResult;

/// As in the CLI, which compares paths: the same molecule twice is a homo-dimer,
/// so the scan reuses one CG body and Manybody accepts it. Names don't matter;
/// other chains make another molecule.
pub fn same_molecule(a: &StructureFile, b: &StructureFile) -> bool {
    (&a.contents, &a.chains) == (&b.contents, &b.chains)
}

/// `grid` is the panel's backend choice; `None` sums exact pairs.
fn build_request(
    mut settings: ScanSettings,
    grid: Option<GridSpec>,
    [a, b]: [StructureFile; 2],
) -> WebScanRequest {
    settings.homo_dimer = same_molecule(&a, &b);
    WebScanRequest {
        mol1_pdb: a.contents,
        mol2_pdb: b.contents,
        chains: [a.chains, b.chains],
        grid,
        settings,
    }
}

/// Run the full scan pipeline synchronously (for native desktop).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_scan_blocking(
    settings: ScanSettings,
    grid: Option<GridSpec>,
    pair: [StructureFile; 2],
    progress: impl Fn(ScanProgress) -> ControlFlow<()>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan(build_request(settings, grid, pair), Some(&progress))
}

/// Run the full scan pipeline asynchronously (for WASM).
#[cfg(target_arch = "wasm32")]
pub async fn run_scan_async(
    settings: ScanSettings,
    grid: Option<GridSpec>,
    pair: [StructureFile; 2],
    progress: impl Fn(ScanProgress) -> ControlFlow<()>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan_async(build_request(settings, grid, pair), Some(&progress)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::ScanInputs;

    /// Picking one structure as both molecules must count as a homo-dimer,
    /// or Manybody refuses the scan; other chains make it another molecule.
    #[test]
    fn same_contents_and_chains_make_a_homo_dimer() {
        let settings = || ScanInputs::default().settings(&Default::default(), &Default::default());
        let file = |name: &str, contents: &[u8], chains: &[&str]| StructureFile {
            name: name.into(),
            contents: contents.to_vec(),
            chains: chains.iter().map(|c| c.to_string()).collect(),
        };
        let homo = |a: StructureFile, b: StructureFile| {
            build_request(settings(), None, [a, b]).settings.homo_dimer
        };
        let pdb = b"ATOM ...";
        assert!(homo(file("a", pdb, &["A"]), file("renamed", pdb, &["A"])));
        assert!(!homo(file("a", pdb, &["A"]), file("a", pdb, &["B"])));
        assert!(!homo(file("a", pdb, &[]), file("a", b"other", &[])));
    }

    /// The panel's grid choice reaches the scan, both ways.
    #[test]
    fn request_carries_the_grid_choice() {
        let settings = ScanInputs::default().settings(&Default::default(), &Default::default());
        let file = || StructureFile {
            name: "a".into(),
            contents: b"ATOM ...".to_vec(),
            chains: vec![],
        };
        for grid in [Some(GridSpec::default()), None] {
            assert_eq!(
                build_request(settings.clone(), grid, [file(), file()]).grid,
                grid
            );
        }
    }
}
