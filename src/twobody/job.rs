//! One background scan, native thread or wasm task, so the main scan and a
//! Solution scan start, stream and stop alike.

use crate::compute::ComputeResult;
use duello::archive::StructureFile;
use duello::web_api::{GridSpec, ScanProgress, ScanSettings};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

pub enum JobEvent {
    /// One finished R-bin.
    Progress(ScanProgress),
    /// The last event; the job is spent after it. Boxed: a scan is large.
    Done(anyhow::Result<Box<ComputeResult>>),
}

pub struct ScanJob {
    result: mpsc::Receiver<anyhow::Result<ComputeResult>>,
    progress: mpsc::Receiver<ScanProgress>,
    /// Checked after every R-bin. One per job, so a new job can't un-cancel an
    /// abandoned one that has yet to reach its next R-bin.
    cancel: Arc<AtomicBool>,
}

impl ScanJob {
    pub fn start(settings: ScanSettings, grid: Option<GridSpec>, pair: [StructureFile; 2]) -> Self {
        let (result_tx, result) = mpsc::channel();
        let (progress_tx, progress) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let on_progress = move |p: ScanProgress| {
            let _ = progress_tx.send(p);
            match stop.load(Ordering::Relaxed) {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let done = crate::compute::run_scan_blocking(settings, grid, pair, on_progress);
            let _ = result_tx.send(done);
        });

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let done = crate::compute::run_scan_async(settings, grid, pair, on_progress).await;
            let _ = result_tx.send(done);
        });

        Self {
            result,
            progress,
            cancel,
        }
    }

    /// What arrived since the last poll, progress first. The result is taken
    /// first: the worker sends all progress before it, so none is left behind.
    pub fn poll(&self) -> Vec<JobEvent> {
        let done = match self.result.try_recv() {
            Ok(done) => Some(done.map(Box::new)),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::anyhow!("Scan thread terminated unexpectedly")))
            }
        };
        let progress = self.progress.try_iter().map(JobEvent::Progress);
        progress.chain(done.map(JobEvent::Done)).collect()
    }

    /// Ends the scan at its next R-bin; drop the job to ignore what follows.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Cancelled jobs still finishing their R-bin. They hold the GPU until then,
/// so no new scan starts before they report back.
#[derive(Default)]
pub struct Retired(Vec<ScanJob>);

impl Retired {
    pub fn retire(&mut self, job: ScanJob) {
        job.cancel();
        self.0.push(job);
    }

    /// Forgets jobs that have ended; their events are ignored.
    pub fn poll(&mut self) {
        let done = |job: &ScanJob| job.poll().iter().any(|e| matches!(e, JobEvent::Done(_)));
        self.0.retain(|job| !done(job));
    }

    pub fn busy(&self) -> bool {
        !self.0.is_empty()
    }
}
