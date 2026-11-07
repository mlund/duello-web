//! The Twobody canvas's windows.

use crate::windows::Slot;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pane {
    FreeEnergy,
    Thermodynamics,
    Poses,
}

impl Pane {
    pub const ALL: [Pane; 3] = [Pane::FreeEnergy, Pane::Thermodynamics, Pane::Poses];
}

impl crate::windows::Pane for Pane {
    const GRID: [usize; 2] = [2, 2];

    fn title(self) -> &'static str {
        match self {
            Pane::FreeEnergy => "Free energy",
            Pane::Thermodynamics => "Thermodynamics",
            Pane::Poses => "Poses",
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
            Pane::Poses => "The lowest-energy states in the scan's table, drawn as beads",
        }
    }

    /// A 2 × 2 grid; the poses are wide.
    fn slot(self) -> Slot {
        let (col, row, span) = match self {
            Pane::FreeEnergy => (0, 0, 1),
            Pane::Thermodynamics => (1, 0, 1),
            Pane::Poses => (0, 1, 2),
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
