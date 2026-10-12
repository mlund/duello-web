//! Records the duello version from `Cargo.lock`, so the app can show which library
//! produced a result. The workspace lockfile is used when building inside faunus-rs.

use std::path::Path;

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let version = ["Cargo.lock", "../Cargo.lock"]
        .iter()
        .map(|lock| Path::new(&dir).join(lock))
        .find_map(|lock| {
            println!("cargo:rerun-if-changed={}", lock.display());
            duello_version(&std::fs::read_to_string(lock).ok()?)
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=DUELLO_VERSION={version}");
}

fn duello_version(lock: &str) -> Option<String> {
    let mut lines = lock.lines();
    lines.find(|line| *line == r#"name = "duello""#)?;
    let version = lines.next()?.strip_prefix("version = ")?;
    Some(version.trim_matches('"').to_string())
}
