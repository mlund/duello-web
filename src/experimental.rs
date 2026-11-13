//! Manybody is not yet validated, so its tab stays greyed out until a tester
//! opens the page with `?manybody`. The browser remembers that;
//! `?manybody=off` forgets it.

/// Whether the page's query string unlocks Manybody, given what this browser
/// remembers, and what to remember from now on (`None`: leave it).
fn resolve(query: &str, remembered: bool) -> (bool, Option<bool>) {
    let asked = query
        .trim_start_matches('?')
        .split('&')
        .find_map(|pair| match pair.split_once('=') {
            Some(("manybody", value)) => Some(value != "off"),
            None if pair == "manybody" => Some(true),
            _ => None,
        });
    match asked {
        Some(unlock) => (unlock, Some(unlock)),
        None => (remembered, None),
    }
}

/// Developers run the native build, so it is always unlocked.
#[cfg(not(target_arch = "wasm32"))]
pub fn manybody_unlocked() -> bool {
    true
}

#[cfg(target_arch = "wasm32")]
pub fn manybody_unlocked() -> bool {
    const KEY: &str = "duello-web.manybody";
    let window = web_sys::window().unwrap();
    let query = window.location().search().unwrap_or_default();
    // Storage can be off (private mode); then the unlock lasts this page load.
    let storage = window.local_storage().ok().flatten();
    let remembered = storage
        .as_ref()
        .and_then(|s| s.get_item(KEY).ok().flatten())
        .is_some();
    let (unlocked, store) = resolve(&query, remembered);
    if let (Some(storage), Some(store)) = (storage, store) {
        let _ = match store {
            true => storage.set_item(KEY, "1"),
            false => storage.remove_item(KEY),
        };
    }
    unlocked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locked_unless_asked_for_or_remembered() {
        assert_eq!(resolve("", false), (false, None));
        assert_eq!(resolve("?tab=1", false), (false, None));
        assert_eq!(resolve("", true), (true, None));
    }

    #[test]
    fn the_query_unlocks_and_is_remembered() {
        assert_eq!(resolve("?manybody", false), (true, Some(true)));
        assert_eq!(resolve("?x=1&manybody=on", false), (true, Some(true)));
    }

    #[test]
    fn off_locks_and_forgets() {
        assert_eq!(resolve("?manybody=off", true), (false, Some(false)));
    }

    #[test]
    fn similar_names_do_not_count() {
        assert_eq!(resolve("?manybodyx", false), (false, None));
        assert_eq!(resolve("?x=manybody", false), (false, None));
    }
}
