//! The Binding Modes window: the poses that carry the scan's B₂ attraction,
//! grouped by similarity and ranked by their contribution, drawn as beads.

use crate::viewer::{charge_color, Rgba, Scene, Sphere, Viewer};
use crate::widgets::placeholder;
use duello::allatom::AllAtom;
use duello::archive::{CgBody, ScanArchive};
use duello::modes::export::{ModeExport, ModeFile};
use duello::modes::{binding_modes, BindingMode, ModeSpec, Modes};
use duello::poses::TablePose;

/// What the pane asks of the Twobody tab, which holds the structures and
/// saves files.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModeSave {
    Csv,
    /// Both bodies' beads of this mode (index into the ranking).
    Cg(usize),
    AllAtom,
}

/// Why the all-atom save is off; shared by the pane and File ▸ Export.
pub const NEEDS_STRUCTURES: &str = "Needs the scanned structures, still loaded in \
     Onebody with the chains they were scanned with";

/// Why there are no modes; shared by the pane and File ▸ Export.
pub const NO_MODES: &str = "No attractive poses: the pair repels in every stored orientation";

impl ModeSave {
    /// Every save, in the order the pane and the menu list them.
    pub fn all(picked: usize) -> [ModeSave; 3] {
        [ModeSave::Csv, ModeSave::Cg(picked), ModeSave::AllAtom]
    }

    pub fn label(self) -> &'static str {
        match self {
            ModeSave::Csv => "Binding modes (.csv)",
            ModeSave::Cg(_) => "Shown mode, CG (.xyz)",
            ModeSave::AllAtom => "Binding modes, all-atom (.pdb)",
        }
    }

    pub fn tip(self) -> &'static str {
        match self {
            ModeSave::Csv => "Rank, share, B₂ contribution, size and pose of every mode",
            ModeSave::Cg(_) => "The shown mode's beads, both molecules",
            ModeSave::AllAtom => "Every mode as one model, all atoms, chains A and B",
        }
    }

    /// Whether it can save: the all-atom one needs the structures.
    pub fn possible(self, all_atom: bool) -> bool {
        all_atom || self != ModeSave::AllAtom
    }

    /// As a button, enabled if `possible`; true when clicked.
    pub fn button(self, ui: &mut egui::Ui, all_atom: bool) -> bool {
        let button = ui.add_enabled(self.possible(all_atom), egui::Button::new(self.label()));
        let button = button
            .on_hover_text(self.tip())
            .on_disabled_hover_text(NEEDS_STRUCTURES);
        button.clicked()
    }
}

/// The ranked modes and the 3D view of the picked one.
#[derive(Default)]
pub struct ModesView {
    /// Ranked once per result, as ranking walks the whole table.
    ranked: Option<Modes>,
    /// A frame showing "Ranking…" has been drawn, so the pause that follows
    /// reads as work rather than a freeze.
    announced: bool,
    picked: usize,
    /// Bumped per result, so the viewer rebuilds its scene for a new one.
    result: u64,
    viewer: Viewer,
}

/// The file `action` saves: the table, mode `i`'s beads, or the all-atom
/// models if the structures were at hand.
pub fn mode_file(action: ModeSave, export: &ModeExport) -> Option<ModeFile> {
    match action {
        ModeSave::Csv => Some(export.csv_file()),
        ModeSave::Cg(i) => export.cg_file(i),
        ModeSave::AllAtom => export.allatom_file(),
    }
}

/// Grouping and count for the pane and its exports alike; the one place to
/// change them.
pub fn spec() -> ModeSpec {
    ModeSpec::default()
}

impl ModesView {
    /// The scan's binding modes, ranked on first use.
    pub fn modes(&mut self, archive: &ScanArchive) -> &Modes {
        self.ranked
            .get_or_insert_with(|| binding_modes(archive, spec()))
    }

    /// `finished` is the scan to rank, None while there is none; `all_atom`
    /// says whether the scanned structures are at hand for a PDB.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        finished: Option<&ScanArchive>,
        all_atom: bool,
    ) -> Option<ModeSave> {
        let Some(archive) = finished else {
            placeholder(ui, "Available once the scan finishes");
            return None;
        };
        if self.ranked.is_none() && !std::mem::replace(&mut self.announced, true) {
            placeholder(ui, "Ranking binding modes…");
            ui.ctx().request_repaint();
            return None;
        }
        let Self {
            ranked,
            picked,
            result,
            viewer,
            ..
        } = self;
        let modes = ranked.get_or_insert_with(|| binding_modes(archive, spec()));
        if modes.modes.is_empty() {
            placeholder(ui, NO_MODES);
            return None;
        }
        picker(ui, &modes.modes, picked);
        let action = saves(ui, *picked, all_atom);
        ui.weak(coverage(modes))
            .on_hover_text(format!("{SHARES_TIP}\n\n{COVERAGE_TIP}"));
        // Looked up after the picker, so this frame draws the pick just made;
        // the viewer caches a scene per key and would keep a stale one.
        let pose = &modes.modes[*picked].pose;
        let bodies = &archive.bodies;
        viewer.ui(ui, (*result, *picked), || scene(pose, bodies));
        action
    }

    /// The shown ranking as files, ranking first if the pane hasn't yet.
    pub fn export<'a>(
        &mut self,
        archive: &'a ScanArchive,
        atoms: Option<[AllAtom; 2]>,
    ) -> ModeExport<'a> {
        ModeExport::with_modes(archive, self.modes(archive).clone(), atoms)
    }

    /// The mode on screen, as an index into the ranking.
    pub fn picked(&self) -> usize {
        self.picked
    }

    /// For a new result: another scene entirely, so ranking, pick and view start over.
    pub fn invalidate(&mut self) {
        self.ranked = None;
        self.announced = false;
        self.picked = 0;
        self.result += 1;
        self.viewer.reset();
    }
}

/// What the two shares mean; part of the picker's and the coverage line's tips.
const SHARES_TIP: &str = "Share of attraction: a mode's part of the summed B₂ \
     contributions of all attractive poses. Share of K_a: its part of B₂ − B₂ʳᵉᶠ, \
     which sets K_a; these can sum past 100 %, as soft repulsion beyond the hard \
     beads offsets part of the attraction, and are absent where the pair doesn't \
     associate.";

const COVERAGE_TIP: &str = "Poses whose B beads lie within the RMSD of a mode's \
     strongest pose join it; a homodimer's swapped view counts as the same pose.";

/// How much of the attraction the listed modes carry.
fn coverage(modes: &Modes) -> String {
    let share: f64 = modes.modes.iter().map(|m| m.share).sum();
    let n = modes.modes.len();
    let noun = if n == 1 { "mode" } else { "modes" };
    let ka = modes.association.map_or(String::new(), |_| {
        let ka: f64 = modes.modes.iter().filter_map(|m| m.association_share).sum();
        format!(" ({:.0}% of K_a)", 100.0 * ka)
    });
    format!(
        "Explains {:.0}% of the attraction{ka} in {n} {noun} (RMSD {} Å)",
        100.0 * share,
        spec().rmsd
    )
}

fn picker(ui: &mut egui::Ui, modes: &[BindingMode], picked: &mut usize) {
    let tip = format!(
        "Binding modes: families of similar poses, ranked by how much of the B₂ \
         attraction they carry, each shown by its strongest pose.\n\n{SHARES_TIP}"
    );
    ui.horizontal(|ui| {
        ui.label("Mode").on_hover_text(&tip);
        egui::ComboBox::from_id_salt("twobody_mode")
            .selected_text(label(*picked, &modes[*picked]))
            .show_ui(ui, |ui| {
                for (k, mode) in modes.iter().enumerate() {
                    ui.selectable_value(picked, k, label(k, mode))
                        .on_hover_text(row_tip(mode));
                }
            })
            .response
            .on_hover_text(tip);
        ui.weak("beads: red −, blue +; neutral sand (1), slate (2)")
            .on_hover_text(
                "Bead charge, full colour at ±1 e. Near-neutral beads are tinted by \
                 molecule: sand for molecule 1, slate for molecule 2.",
            );
    });
}

fn label(k: usize, mode: &BindingMode) -> String {
    let ka = mode
        .association_share
        .map_or(String::new(), |s| format!(" · {:.0}% of K_a", 100.0 * s));
    let n = mode.n_poses;
    let noun = if n == 1 { "pose" } else { "poses" };
    format!(
        "#{}  {:.0}% of attraction{ka} · {n} {noun} · U {:.1} kT · R {:.1} Å",
        k + 1,
        100.0 * mode.share,
        mode.pose.energy,
        mode.pose.r
    )
}

fn row_tip(mode: &BindingMode) -> String {
    format!(
        "Contributes {:.3e} Å³ to B₂ over its tabulated poses (orientations at \
         each separation); U and R are its strongest pose's, at ω = {:.0}°",
        mode.b2,
        mode.pose.omega.to_degrees()
    )
}

/// The pane's save buttons; the tab writes the files.
fn saves(ui: &mut egui::Ui, picked: usize, all_atom: bool) -> Option<ModeSave> {
    ui.horizontal(|ui| {
        let clicked = ModeSave::all(picked).map(|save| save.button(ui, all_atom).then_some(save));
        clicked.into_iter().flatten().next()
    })
    .inner
}

/// Each molecule's neutral bead colour, warm and cool around the charge map's
/// grey, so the two bodies tell apart while red and blue still read as charge.
const NEUTRAL_TINTS: [Rgba; 2] = [[0.84, 0.78, 0.66, 1.0], [0.66, 0.74, 0.86, 1.0]];

/// The charge map with `neutral` in place of its grey; fades out as |z| → 1, so
/// a full charge looks the same in either molecule.
fn tinted_charge_color(z: f64, neutral: Rgba) -> Rgba {
    let end = charge_color(z.signum());
    let t = z.abs().min(1.0) as f32;
    std::array::from_fn(|i| (1.0 - t) * neutral[i] + t * end[i])
}

/// Both bodies as beads coloured by charge, framed about their joint centre.
fn scene(pose: &TablePose, bodies: &[CgBody; 2]) -> Scene {
    let placed = pose.place(bodies);
    let spheres: Vec<Sphere> = (placed.iter().zip(bodies).zip(NEUTRAL_TINTS))
        .flat_map(|((beads, body), neutral)| {
            (beads.iter().zip(&body.radii).zip(&body.charges)).map(move |((p, &r), &z)| Sphere {
                center: p.map(|x| x as f32),
                radius: r as f32,
                color: tinted_charge_color(z, neutral),
            })
        })
        .collect();
    let bounds = bounds(&spheres);
    Scene {
        spheres,
        bounds,
        ..Default::default()
    }
}

fn bounds(spheres: &[Sphere]) -> crate::viewer::Bounds {
    let n = spheres.len().max(1) as f32;
    let centre = spheres.iter().fold([0.0; 3], |c, s| {
        std::array::from_fn(|i| c[i] + s.center[i] / n)
    });
    let reach = |s: &Sphere| {
        let d: f32 = (0..3).map(|i| (s.center[i] - centre[i]).powi(2)).sum();
        d.sqrt() + s.radius
    };
    (centre, spheres.iter().map(reach).fold(1.0, f32::max))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each save writes duello's own file for it, from the ranking the pane
    /// already holds: the same bytes as `duello modes` and session extracts.
    #[test]
    fn each_save_writes_duellos_file_from_the_shown_ranking() {
        use duello::modes::export::ModeExport;
        let archive = crate::test_archive::attractive();
        let mut view = ModesView::default();
        let export = view.export(&archive, None);
        let fresh = ModeExport::new(&archive, spec(), None);
        let name = |a| mode_file(a, &export).map(|f| f.name);
        assert_eq!(name(ModeSave::Csv), Some("modes.csv".into()));
        assert_eq!(name(ModeSave::Cg(1)), Some("mode02_cg.xyz".into()));
        assert_eq!(name(ModeSave::AllAtom), None); // no structures
        assert_eq!(
            mode_file(ModeSave::Csv, &export).unwrap().contents,
            fresh.csv()
        );
    }

    /// Shares set by hand, so the wording is checked against literals.
    fn two_modes(association: bool) -> Modes {
        let mut modes = binding_modes(&crate::test_archive::attractive(), spec());
        modes.modes.truncate(2);
        assert_eq!(modes.modes.len(), 2, "the fixture should give two modes");
        modes.association = association.then_some(-1.0);
        for (mode, (share, ka)) in modes.modes.iter_mut().zip([(0.42, 0.5), (0.45, 0.55)]) {
            mode.share = share;
            mode.association_share = association.then_some(ka);
        }
        modes
    }

    #[test]
    fn coverage_and_labels_give_both_shares() {
        let modes = two_modes(true);
        assert_eq!(
            coverage(&modes),
            "Explains 87% of the attraction (105% of K_a) in 2 modes (RMSD 4 Å)"
        );
        let first = label(0, &modes.modes[0]);
        let poses = format!(
            "#1  42% of attraction · 50% of K_a · {} poses · U ",
            modes.modes[0].n_poses
        );
        assert!(first.starts_with(&poses), "{first}");
    }

    /// Without association there is no K_a to share.
    #[test]
    fn without_association_only_the_attraction_share_shows() {
        let modes = two_modes(false);
        assert_eq!(
            coverage(&modes),
            "Explains 87% of the attraction in 2 modes (RMSD 4 Å)"
        );
        let first = label(0, &modes.modes[0]);
        let poses = format!(
            "#1  42% of attraction · {} poses · U ",
            modes.modes[0].n_poses
        );
        assert!(first.starts_with(&poses), "{first}");
    }

    #[test]
    fn the_pane_renders_with_without_and_before_modes() {
        let (attractive, repulsive) = (
            crate::test_archive::attractive(),
            crate::test_archive::archive(),
        );
        for (archive, all_atom) in [
            (Some(&attractive), true),
            (Some(&repulsive), false),
            (None, false),
        ] {
            let mut view = ModesView::default();
            // The first frame announces the ranking, the second shows it.
            for _ in 0..2 {
                crate::widgets::headless(|ui| {
                    assert_eq!(view.ui(ui, archive, all_atom), None);
                });
            }
        }
        assert!(binding_modes(&repulsive, spec()).modes.is_empty());
    }

    /// One ranking for the pane, the exports and the CLI: duello's.
    #[test]
    fn the_view_ranks_the_scans_binding_modes() {
        let archive = crate::test_archive::attractive();
        let want = binding_modes(&archive, spec());
        assert!(!want.modes.is_empty(), "the fixture should attract");
        let mut view = ModesView::default();
        let got: Vec<f64> = view.modes(&archive).modes.iter().map(|m| m.b2).collect();
        assert_eq!(got, want.modes.iter().map(|m| m.b2).collect::<Vec<_>>());
    }

    #[test]
    fn neutral_beads_take_their_molecules_tint_and_full_charges_do_not() {
        let [a, b] = NEUTRAL_TINTS;
        assert_eq!(tinted_charge_color(0.0, a), a);
        assert_eq!(tinted_charge_color(0.0, b), b);
        for z in [-1.0, 1.0, 2.0] {
            assert_eq!(tinted_charge_color(z, a), charge_color(z));
            assert_eq!(tinted_charge_color(z, b), charge_color(z));
        }
    }
}
