//! What the panes show and Save writes: the live run, or one opened from a
//! session. Borrowed, so a pane computes only what it draws and nothing is
//! copied per frame.

use duello::nbody::{NBodyRun, RdfCurve, RunRecord, RunStats, ScatteringCurve};
use std::borrow::Cow;

/// A finished run from a session file; it can be shown and saved again.
pub struct Opened {
    pub record: RunRecord,
    pub trajectory: Option<Vec<u8>>,
}

#[derive(Clone, Copy)]
pub enum Results<'a> {
    Live(&'a NBodyRun),
    Opened(&'a Opened),
}

impl<'a> Results<'a> {
    pub fn of(run: &'a Option<NBodyRun>, opened: &'a Option<Opened>) -> Option<Self> {
        run.as_ref()
            .map(Self::Live)
            .or(opened.as_ref().map(Self::Opened))
    }

    pub fn curve(self) -> Cow<'a, ScatteringCurve> {
        match self {
            Self::Live(run) => Cow::Owned(run.curve()),
            Self::Opened(o) => Cow::Borrowed(&o.record.curve),
        }
    }

    pub fn rdf(self) -> Cow<'a, RdfCurve> {
        match self {
            Self::Live(run) => Cow::Owned(run.pair_correlation()),
            Self::Opened(o) => Cow::Borrowed(&o.record.rdf),
        }
    }

    /// (sweep, U/N in kT).
    pub fn energy(self) -> &'a [[f64; 2]] {
        match self {
            Self::Live(run) => run.energy_trace(),
            Self::Opened(o) => &o.record.energy,
        }
    }

    pub fn stats(self) -> Cow<'a, RunStats> {
        match self {
            Self::Live(run) => Cow::Owned(run.stats()),
            Self::Opened(o) => Cow::Borrowed(&o.record.stats),
        }
    }

    /// None if not recorded, or not saved with an opened run. A live one is
    /// borrowed from the engine, as it can reach hundreds of MB.
    pub fn trajectory(self) -> Option<Cow<'a, [u8]>> {
        match self {
            Self::Live(run) => run.trajectory_bytes().ok(),
            Self::Opened(o) => o.trajectory.as_deref().map(Cow::Borrowed),
        }
    }

    /// Curve and statistics as CSV.
    pub fn report_csv(self) -> String {
        match self {
            Self::Live(run) => run.report_csv(),
            Self::Opened(o) => o.record.report_csv(),
        }
    }

    /// The record to save.
    pub fn record(self) -> Cow<'a, RunRecord> {
        match self {
            Self::Live(run) => Cow::Owned(run.results()),
            Self::Opened(o) => Cow::Borrowed(&o.record),
        }
    }
}
