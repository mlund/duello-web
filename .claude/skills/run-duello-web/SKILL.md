---
name: run-duello-web
description: Launch duello-web (trunk) and drive it in headless Chrome to check a GUI change in the real app - click, type, upload, screenshot, read console errors. Use when asked to run, screenshot or browser-test duello-web.
---

# Run duello-web

egui draws the whole app on one `<canvas>`: there is no DOM to query.
Drive it by **coordinates**, and verify by **reading screenshots**.

## 1. Serve

From `duello-web/` (skip if port 8080 already answers; trunk rebuilds on save):

```bash
trunk serve --release --port 8080      # run in background
until curl -sf localhost:8080 >/dev/null; do sleep 1; done
```

Stop: `lsof -ti:8080 -sTCP:LISTEN | xargs kill`.

## 2. Driver

Once per checkout, in this skill's directory:

```bash
npm i        # Playwright only; it uses the installed Google Chrome
```

Start it in the background from a scratch directory (outputs land in
`$DRIVER_OUT`, default `./run-out`):

```bash
DRIVER_OUT=<scratch>/run-out node <this dir>/driver.mjs   # prints "ready on :9333"
```

Each step is one curl; ops are listed at the top of `driver.mjs`:

```bash
D=localhost:9333/cmd; c(){ curl -s "$D?$1"; echo; }
c "op=nav&url=http://localhost:8080"; c "op=wait&ms=4000"
c "op=shot&name=01"        # prints the PNG path: Read it
c "op=logs"                # console errors/warnings + downloads since last call
c "op=quit"
```

## 3. Representative path: fetch a PDB and see Onebody

Viewport 1600×1000, light theme, fresh page. Coordinates move when the
layout does (more structures, chain rows) - re-screenshot before clicking.

```bash
c "op=click&x=160&y=106&n=2"; c "op=key&key=Meta+a"; c "op=type&text=1BRS"
c "op=click&x=218&y=106"      # Fetch
c "op=wait&ms=8000"; c "op=shot&name=fetched"
```

Expect: titration curve, multipole table, surface viewer, Atoms table and,
for multi-chain files, a `Chains` row of checkboxes.

Landmarks on a fresh page: `File` (20,39), tabs `Onebody` (87,39),
`Twobody` (156,39), `Manybody` (230,39); in the File menu `Open session…`
(65,66), `Save session…` (65,89). Twobody `Run` is at (62,334) until a
finished scan widens the left panel.

## Gotchas

- **Hover tooltips**: `op=move`, then `op=wait&ms=1500` before the shot.
- **Open session**: `op=upload&path=<zip>` first, then File → Open session….
  Headless Chrome shows rfd's fallback dialog: click its `Ok` (800,498).
- **Save session / exports**: no dialog; the file lands in
  `$DRIVER_OUT/downloads/`, reported by `op=logs`. Inspect with `unzip -p`.
- **Scans and titration take seconds**: wait (8 s fetch, ~20 s for a small
  Twobody scan), then screenshot; a half-drawn pane means wait longer.
- **Duplicate structures** carry their kept chains in pickers, e.g.
  `1BRS.pdb (A)`; the Onebody list still shows only the name.
- **Restarting the driver**: `op=quit` answers before Chrome closes; wait
  until `lsof -ti:9333 -sTCP:LISTEN` is empty before starting another.
- **Trunk watches this directory too**: `npm i` here triggers one harmless
  rebuild; wait for `success` in trunk's log before navigating.
- `npx playwright install` is not needed and is slow; `channel: 'chrome'`
  uses the system browser.
