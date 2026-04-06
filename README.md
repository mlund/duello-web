# Duello Web

Browser-based GUI for [Duello](https://github.com/mlund/duello) -- compute osmotic second virial coefficients (B2) and dissociation constants (Kd) for rigid macromolecules using WebGPU.

[![Run in Browser](https://img.shields.io/badge/Run_in_Browser!-WebGPU-orange?style=for-the-badge&logo=webassembly)](https://mlund.github.io/duello-web/)

## Run in Browser

Visit **https://mlund.github.io/duello-web/** (requires a WebGPU-capable browser: Chrome 113+, Edge 113+, or Safari 18+).

## Run Locally

### Native Desktop App

```sh
cargo run --release
```

Runs as a native desktop application with GPU acceleration.

### WASM Development Server

Requires [Trunk](https://trunkrs.dev/):

```sh
cargo install trunk
rustup target add wasm32-unknown-unknown
trunk serve
```

Opens at http://127.0.0.1:8080/.

## Features

- Upload PDB files or fetch directly from RCSB by PDB ID
- Coarse-grain with CALVADOS 3, Kim-Hummer, or Pasquier force fields
- GPU-accelerated 6D angular scan via WebGPU (browser) or wgpu (native)
- Interactive PMF plot with free energy and average energy
- Reports B2, reduced B2, Kd, molecular weights, and net charges
- Download CG structures (.xyz), topology (.yaml), and PMF data (.csv)
