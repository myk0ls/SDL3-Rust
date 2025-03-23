use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Get the output directory (where the executable will be built)
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir)
        .parent().unwrap()  // target/debug/build/your_project-xxxx/
        .parent().unwrap()  // target/debug/
        .parent().unwrap(); // target/

    // Copy assets
    fs::create_dir_all(dest_path.join("assets")).unwrap();
    fs::copy("assets/texture.bmp", dest_path.join("assets/texture.bmp")).unwrap();

    // Copy shaders
    fs::create_dir_all(dest_path.join("shaders")).unwrap();
    fs::copy("shaders/cube-texture.vert.spv", dest_path.join("shaders/cube-texture.vert.spv")).unwrap();
    fs::copy("shaders/cube-texture.frag.spv", dest_path.join("shaders/cube-texture.frag.spv")).unwrap();
}