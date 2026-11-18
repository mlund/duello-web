//! The Twobody canvas's windows.

use crate::windows::Slot;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pane {
    FreeEnergy,
    Thermodynamics,
    SolutionScan,
    Modes,
}

impl Pane {
    pub const ALL: [Pane; 4] = [
        Pane::FreeEnergy,
        Pane::Thermodynamics,
        Pane::SolutionScan,
        Pane::Modes,
    ];
}

impl crate::windows::Pane for Pane {
    const GRID: [usize; 2] = [3, 2];

    fn title(self) -> &'static str {
        match self {
            Pane::FreeEnergy => "Free energy",
            Pane::Thermodynamics => "Thermodynamics",
            Pane::SolutionScan => "Solution scan",
            Pane::Modes => "Binding Modes",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Pane::FreeEnergy => {
                "Angle-averaged curves against the mass-centre separation R; \
                 pick one with Show"
            }
            Pane::Thermodynamics => {
                "Second virial coefficient, Kirkwood–Buff integral and, where the pair \
                 associates, its standard free energy split"
            }
            Pane::SolutionScan => {
                "B₂ along pH, salt or temperature: one full scan per point, the other \
                 conditions from Onebody. Not saved in the session; save the CSV to keep it."
            }
            Pane::Modes => {
                "The poses that carry the B₂ attraction: similar poses grouped, ranked by \
                 their contribution, drawn as beads"
            }
        }
    }

    /// A 3 × 2 grid; the poses are wide.
    fn slot(self) -> Slot {
        let (col, row, span) = match self {
            Pane::FreeEnergy => (0, 0, 1),
            Pane::Thermodynamics => (1, 0, 1),
            Pane::SolutionScan => (2, 0, 1),
            Pane::Modes => (0, 1, 3),
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

    /// Three analysis panes on top, the poses wide below.
    #[test]
    fn the_solution_scan_sits_beside_thermodynamics() {
        use crate::windows::Pane as _;
        let slot = super::Pane::SolutionScan.slot();
        assert_eq!((slot.col, slot.row, slot.span), (2, 0, 1));
        assert_eq!(super::Pane::Modes.slot().span, 3);
    }
}
