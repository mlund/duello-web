//! Bead tables: sortable, resizable, striped.

use duello::molecule::BeadRow;
use std::cmp::Ordering;

/// A table column over [`BeadRow`]s; `Index` is the bead's place in the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Column {
    Index,
    Chain,
    Residue,
    Kind,
    Charge,
    Variance,
    Mass,
    Sigma,
    Epsilon,
    Lambda,
}

/// Orders `rows`, indices into `beads`, by `column`; ties keep their order.
pub fn sort_rows(beads: &[BeadRow], rows: &mut [usize], column: Column, ascending: bool) {
    rows.sort_by(|&a, &b| {
        let order = column.compare((a, &beads[a]), (b, &beads[b]));
        if ascending {
            order
        } else {
            order.reverse()
        }
    });
}

/// Sort state and row order of one table; rows are re-sorted only when the
/// molecule or the sort changes.
pub struct BeadTable {
    id: &'static str,
    columns: &'static [Column],
    /// Which beads it lists.
    keep: fn(&BeadRow) -> bool,
    sort: (Column, bool),
    order: Vec<usize>,
    /// Molecule `order` was built for.
    sorted_for: Option<u64>,
}

impl BeadTable {
    pub fn new(id: &'static str, columns: &'static [Column], keep: fn(&BeadRow) -> bool) -> Self {
        Self {
            id,
            columns,
            keep,
            sort: (Column::Index, true),
            order: Vec::new(),
            sorted_for: None,
        }
    }

    /// `molecule` identifies `beads`, so a new titration re-sorts.
    pub fn ui(&mut self, ui: &mut egui::Ui, beads: &[BeadRow], molecule: u64) {
        if self.sorted_for != Some(molecule) {
            self.order = (0..beads.len())
                .filter(|&i| (self.keep)(&beads[i]))
                .collect();
            self.sorted_for = Some(molecule);
            self.resort(beads);
        }
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        let clicked = std::cell::Cell::new(None);
        egui_extras::TableBuilder::new(ui)
            .id_salt(self.id)
            .striped(true)
            .resizable(true)
            .columns(
                egui_extras::Column::auto().at_least(36.0),
                self.columns.len(),
            )
            .header(row_height + 2.0, |mut header| {
                for &column in self.columns {
                    header.col(|ui| {
                        if self.header_cell(ui, column) {
                            clicked.set(Some(column));
                        }
                    });
                }
            })
            .body(|body| {
                body.rows(row_height, self.order.len(), |mut row| {
                    let i = self.order[row.index()];
                    for &column in self.columns {
                        row.col(|ui| {
                            ui.label(column.text(i, &beads[i]));
                        });
                    }
                })
            });
        if let Some(column) = clicked.get() {
            // Clicking the sorted column flips it; another sorts ascending.
            self.sort = (column, self.sort != (column, true));
            self.resort(beads);
        }
    }

    /// True when clicked.
    fn header_cell(&self, ui: &mut egui::Ui, column: Column) -> bool {
        let arrow = match self.sort {
            (c, true) if c == column => " ▲",
            (c, false) if c == column => " ▼",
            _ => "",
        };
        ui.selectable_label(!arrow.is_empty(), format!("{}{arrow}", column.label()))
            .on_hover_text(format!("{}\n\nClick to sort.", column.tooltip()))
            .clicked()
    }

    fn resort(&mut self, beads: &[BeadRow]) {
        let (column, ascending) = self.sort;
        sort_rows(beads, &mut self.order, column, ascending);
    }
}

impl Column {
    fn label(self) -> &'static str {
        match self {
            Column::Index => "#",
            Column::Chain => "Chain",
            Column::Residue => "Res",
            Column::Kind => "Kind",
            Column::Charge => "⟨z⟩",
            Column::Variance => "Var(z)",
            Column::Mass => "Mass",
            Column::Sigma => "σ",
            Column::Epsilon => "ε",
            Column::Lambda => "λ",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Column::Index => "Bead number in the coarse-grained body",
            Column::Chain => "Chain ID",
            Column::Residue => "Residue number",
            Column::Kind => "Atom kind: the residue, or a site such as NTR or CTR",
            Column::Charge => "Mean charge from the titration, e",
            Column::Variance => {
                "Charge fluctuation ⟨z²⟩ − ⟨z⟩², e²; blank for beads that don't titrate"
            }
            Column::Mass => "Mass, g/mol",
            Column::Sigma => "Bead diameter of the short-range potential, Å",
            Column::Epsilon => "Short-range well depth, kJ/mol",
            Column::Lambda => "Hydrophobicity (Ashbaugh–Hatch λ)",
        }
    }

    fn text(self, i: usize, bead: &BeadRow) -> String {
        let optional =
            |x: Option<f64>, digits: usize| x.map_or(String::new(), |v| format!("{v:.digits$}"));
        match self {
            Column::Index => (i + 1).to_string(),
            Column::Chain => bead.chain.clone(),
            Column::Residue => bead.residue.to_string(),
            Column::Kind => bead.kind.clone(),
            Column::Charge => format!("{:+.2}", bead.charge),
            Column::Variance => optional(bead.charge_variance, 2),
            Column::Mass => format!("{:.1}", bead.mass),
            Column::Sigma => optional(bead.sigma, 2),
            Column::Epsilon => optional(bead.epsilon, 3),
            Column::Lambda => optional(bead.lambda, 3),
        }
    }

    fn compare(self, (i, a): (usize, &BeadRow), (j, b): (usize, &BeadRow)) -> Ordering {
        // `None` before any value.
        let optional = |x: Option<f64>, y: Option<f64>| match (x, y) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            _ => x.is_some().cmp(&y.is_some()),
        };
        match self {
            Column::Index => i.cmp(&j),
            Column::Chain => a.chain.cmp(&b.chain),
            Column::Residue => a.residue.cmp(&b.residue),
            Column::Kind => a.kind.cmp(&b.kind),
            Column::Charge => a.charge.total_cmp(&b.charge),
            Column::Variance => optional(a.charge_variance, b.charge_variance),
            Column::Mass => a.mass.total_cmp(&b.mass),
            Column::Sigma => optional(a.sigma, b.sigma),
            Column::Epsilon => optional(a.epsilon, b.epsilon),
            Column::Lambda => optional(a.lambda, b.lambda),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bead(chain: &str, residue: i32, charge: f64, sigma: Option<f64>) -> BeadRow {
        BeadRow {
            chain: chain.into(),
            residue,
            kind: "X".into(),
            charge,
            charge_variance: None,
            mass: 100.0,
            sigma,
            epsilon: None,
            lambda: None,
        }
    }

    fn sorted(beads: &[BeadRow], column: Column, ascending: bool) -> Vec<usize> {
        let mut rows: Vec<usize> = (0..beads.len()).collect();
        sort_rows(beads, &mut rows, column, ascending);
        rows
    }

    /// Numbers sort as numbers: residue 10 after 9, not between 1 and 2.
    #[test]
    fn residues_sort_numerically_both_ways() {
        let beads = [
            bead("A", 10, 0.0, None),
            bead("A", 9, 0.0, None),
            bead("A", 100, 0.0, None),
        ];
        assert_eq!(sorted(&beads, Column::Residue, true), [1, 0, 2]);
        assert_eq!(sorted(&beads, Column::Residue, false), [2, 0, 1]);
    }

    #[test]
    fn ties_keep_their_order_and_missing_values_come_first() {
        let beads = [
            bead("B", 1, 1.0, Some(6.0)),
            bead("A", 2, -1.0, None),
            bead("B", 3, 1.0, Some(5.0)),
        ];
        assert_eq!(sorted(&beads, Column::Chain, true), [1, 0, 2]);
        assert_eq!(sorted(&beads, Column::Charge, false), [0, 2, 1]);
        assert_eq!(sorted(&beads, Column::Sigma, true), [1, 2, 0]);
        // A subset sorts within itself.
        let mut rows = vec![2, 0];
        sort_rows(&beads, &mut rows, Column::Index, true);
        assert_eq!(rows, [0, 2]);
    }
}
