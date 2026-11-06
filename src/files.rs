//! Saving files: a dialog natively, a download in the browser.

/// Native: a save dialog (cancel is not an error). Browser: a download.
pub fn save_file(content: &[u8], filename: &str) -> std::io::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = rfd::FileDialog::new().set_file_name(filename).save_file() {
        std::fs::write(path, content)?;
    }
    #[cfg(target_arch = "wasm32")]
    download_bytes_as_file(content, filename);
    Ok(())
}

/// Trigger a browser file download (WASM only).
#[cfg(target_arch = "wasm32")]
pub fn download_bytes_as_file(content: &[u8], filename: &str) {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let blob = web_sys::Blob::new_with_u8_array_sequence(&js_sys::Array::of1(
        &js_sys::Uint8Array::from(content),
    ))
    .unwrap();
    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();
    let a: web_sys::HtmlAnchorElement = document.create_element("a").unwrap().dyn_into().unwrap();
    a.set_href(&url);
    a.set_download(filename);
    a.click();
    web_sys::Url::revoke_object_url(&url).unwrap();
}
