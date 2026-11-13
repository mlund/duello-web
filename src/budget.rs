//! Work per UI frame: long computations step a little each frame, sized so the
//! page stays responsive while they make progress.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;

/// Wall time per UI frame spent stepping.
const FRAME_BUDGET_MS: f64 = 30.0;

/// Steps for the next frame, given that `steps` took since `start`. At most
/// doubles, so one fast frame can't make the next one stall.
pub fn retune(steps: u64, start: Instant) -> u64 {
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let scale = (FRAME_BUDGET_MS / ms.max(0.1)).clamp(0.1, 2.0);
    ((steps as f64 * scale) as u64).max(1)
}
