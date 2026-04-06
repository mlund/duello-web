use crate::app::{CgPolicy, ScanInputs};
use duello::web_api::{WebScanRequest, WebScanResult};
#[cfg(target_arch = "wasm32")]
use std::sync::{atomic::AtomicBool, Arc};

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
        ph: inputs.ph,
        model: inputs.model.name().to_string(),
        single_bead: inputs.cg == CgPolicy::Single,
        rmin: inputs.rmin,
        rmax: inputs.rmax,
        dr: inputs.dr,
        molarity: inputs.molarity,
        temperature: inputs.temperature,
        max_ndiv: inputs.max_ndiv,
        gradient_threshold: inputs.gradient_threshold,
        cutoff: inputs.cutoff,
        grid_size: inputs.grid_size,
        homo_dimer: inputs.homo_dimer,
    }
}

/// Run the full scan pipeline synchronously (for native desktop).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_scan_blocking(
    inputs: &ScanInputs,
    mol1_pdb: Vec<u8>,
    mol2_pdb: Vec<u8>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan(build_request(inputs, mol1_pdb, mol2_pdb))
}

/// Run the full scan pipeline asynchronously (for WASM).
#[cfg(target_arch = "wasm32")]
pub async fn run_scan_async(
    inputs: ScanInputs,
    mol1_pdb: Vec<u8>,
    mol2_pdb: Vec<u8>,
    progress: impl Fn(usize, usize),
    cancel: Arc<AtomicBool>,
) -> anyhow::Result<ComputeResult> {
    duello::web_api::run_scan_async(
        build_request(&inputs, mol1_pdb, mol2_pdb),
        Some(&progress),
        Some(&cancel),
    )
    .await
}
