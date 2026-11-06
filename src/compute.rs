use crate::app::ScanInputs;
use duello::web_api::{ScanProgress, WebScanRequest, WebScanResult};
use std::ops::ControlFlow;

/// Status of the computation.
pub enum ComputeStatus {
    Idle,
    Running { current_r: usize, total_r: usize },
}

pub type ComputeResult = WebScanResult;

/// Build a scan request from UI inputs and PDB byte data.
fn build_request(inputs: &ScanInputs, mol1_pdb: Vec<u8>, mol2_pdb: Vec<u8>) -> WebScanRequest {
    let mut settings: duello::archive::ScanSettings = inputs.into();
    // As in the CLI, which compares paths: the same structure twice is a
    // homo-dimer, so the scan reuses one CG body and Manybody accepts it.
    settings.homo_dimer |= mol1_pdb == mol2_pdb;
    WebScanRequest {
        mol1_pdb,
        mol2_pdb,
        settings,
    }
}

/// Run the full scan pipeline synchronously (for native desktop).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_scan_blocking(
    inputs: &ScanInputs,
    mol1_pdb: Vec<u8>,
    mol2_pdb: Vec<u8>,
    progress: impl Fn(ScanProgress) -> ControlFlow<()>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan(build_request(inputs, mol1_pdb, mol2_pdb), Some(&progress))
}

/// Run the full scan pipeline asynchronously (for WASM).
#[cfg(target_arch = "wasm32")]
pub async fn run_scan_async(
    inputs: ScanInputs,
    mol1_pdb: Vec<u8>,
    mol2_pdb: Vec<u8>,
    progress: impl Fn(ScanProgress) -> ControlFlow<()>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan_async(build_request(&inputs, mol1_pdb, mol2_pdb), Some(&progress))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use duello::Subdivision;

    /// The UI has no homo-dimer switch, so loading one PDB twice must still
    /// count as a homo-dimer, or Manybody refuses the scan.
    #[test]
    fn identical_structures_make_a_homo_dimer() {
        let inputs = ScanInputs::default();
        let pdb = b"ATOM ...".to_vec();
        assert!(
            build_request(&inputs, pdb.clone(), pdb.clone())
                .settings
                .homo_dimer
        );
        assert!(
            !build_request(&inputs, pdb, b"other".to_vec())
                .settings
                .homo_dimer
        );
    }

    #[test]
    fn request_carries_the_chosen_subdivision() {
        let inputs = ScanInputs {
            subdivision: Subdivision::Lattice,
            ..Default::default()
        };
        let request = build_request(&inputs, vec![], vec![]);
        assert_eq!(request.settings.subdivision, Subdivision::Lattice);
    }
}
