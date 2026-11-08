use duello::archive::StructureFile;
use duello::web_api::{ScanProgress, ScanSettings, WebScanRequest, WebScanResult};
use std::ops::ControlFlow;

/// Status of the computation.
pub enum ComputeStatus {
    Idle,
    Running { current_r: usize, total_r: usize },
}

pub type ComputeResult = WebScanResult;

fn build_request(mut settings: ScanSettings, [a, b]: [StructureFile; 2]) -> WebScanRequest {
    // As in the CLI, which compares paths: the same molecule twice is a
    // homo-dimer, so the scan reuses one CG body and Manybody accepts it.
    // Names don't matter; other chains make another molecule.
    settings.homo_dimer = (&a.contents, &a.chains) == (&b.contents, &b.chains);
    WebScanRequest {
        mol1_pdb: a.contents,
        mol2_pdb: b.contents,
        chains: [a.chains, b.chains],
        settings,
    }
}

/// Run the full scan pipeline synchronously (for native desktop).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_scan_blocking(
    settings: ScanSettings,
    pair: [StructureFile; 2],
    progress: impl Fn(ScanProgress) -> ControlFlow<()>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan(build_request(settings, pair), Some(&progress))
}

/// Run the full scan pipeline asynchronously (for WASM).
#[cfg(target_arch = "wasm32")]
pub async fn run_scan_async(
    settings: ScanSettings,
    pair: [StructureFile; 2],
    progress: impl Fn(ScanProgress) -> ControlFlow<()>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan_async(build_request(settings, pair), Some(&progress)).await
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
            build_request(settings(), [a, b]).settings.homo_dimer
        };
        let pdb = b"ATOM ...";
        assert!(homo(file("a", pdb, &["A"]), file("renamed", pdb, &["A"])));
        assert!(!homo(file("a", pdb, &["A"]), file("a", pdb, &["B"])));
        assert!(!homo(file("a", pdb, &[]), file("a", b"other", &[])));
    }
}
