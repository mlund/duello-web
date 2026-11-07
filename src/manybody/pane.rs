//! The Manybody canvas's windows.

use crate::windows::Slot;

#[derive(Clone, Copy, PartialEq, Debug)]
/// Declared in tiling order, as `ALL` lists them.
pub enum Pane {
    Simulation,
    Scattering,
    PairCorrelation,
    Energy,
    Stats,
}

impl Pane {
    pub const ALL: [Pane; 5] = [
        Pane::Simulation,
        Pane::Scattering,
        Pane::PairCorrelation,
        Pane::Energy,
        Pane::Stats,
    ];
}

impl crate::windows::Pane for Pane {
    const GRID: [usize; 2] = [3, 2];

    fn title(self) -> &'static str {
        match self {
            Pane::Energy => "Energy",
            Pane::Scattering => "Scattering",
            Pane::PairCorrelation => "g(r)",
            Pane::Stats => "MC stats",
            Pane::Simulation => "Simulation",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Pane::Energy => {
                "Energy per molecule, U/N: through equilibration, then afresh from the \
                 start of production"
            }
            Pane::Scattering => "Structure factor and, with Detailed I(q), I(q) and P(q)",
            Pane::PairCorrelation => "Radial distribution of mass centres, g(r)",
            Pane::Stats => "Run state, move acceptance, energy statistics and downloads",
            Pane::Simulation => "3D view of the simulation box",
        }
    }

    /// A 3 × 2 grid; the stats are wide.
    fn slot(self) -> Slot {
        let (col, row, span) = match self {
            Pane::Simulation => (0, 0, 1),
            Pane::Scattering => (1, 0, 1),
            Pane::PairCorrelation => (2, 0, 1),
            Pane::Energy => (0, 1, 1),
            Pane::Stats => (1, 1, 2),
        };
        Slot { col, row, span }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn panes_tile_the_canvas() {
        crate::windows::assert_tiled(&super::Pane::ALL);
    }
}
