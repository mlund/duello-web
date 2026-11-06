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
    WebScanRequest {
        mol1_pdb,
        mol2_pdb,
        settings: inputs.into(),
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
