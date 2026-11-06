mod app;
mod compute;
mod manybody;
#[cfg(not(target_arch = "wasm32"))]
mod version;

/// The 3D viewer's impostor shaders depth-test, so the surface needs a depth buffer.
const DEPTH_BITS: u8 = 24;

fn main() -> eframe::Result {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 720.0]),
            renderer: eframe::Renderer::Wgpu,
            depth_buffer: DEPTH_BITS,
            ..Default::default()
        };
        eframe::run_native(
            "DuelloWeb",
            native_options,
            Box::new(|cc| Ok(Box::new(app::DuelloApp::new(cc)))),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).expect("Failed to init console_log");

        let web_options = eframe::WebOptions {
            depth_buffer: DEPTH_BITS,
            ..Default::default()
        };
        wasm_bindgen_futures::spawn_local(async {
            let document = web_sys::window().unwrap().document().unwrap();
            let canvas = document
                .get_element_by_id("duello_canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            let start_result = eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| Ok(Box::new(app::DuelloApp::new(cc)))),
                )
                .await;
            if let Err(e) = start_result {
                log::error!("Failed to start eframe: {e:?}");
            }
        });
        Ok(())
    }
}
