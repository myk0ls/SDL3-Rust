//use std::{fmt::Error, io::{BufReader, Cursor, Error}, primitive};
//use gltf::{image, json::Error, Gltf, Mesh, Texture};

use easy_gltf::model::Mode;

/*
pub struct Model<'a> {
    pub meshes: Vec<Mesh<'a>>,
    pub textures: Vec<Texture<'a>>
}
 */

 

pub fn load_gltf(path: &str){
    let scenes = easy_gltf::load(path).expect("Failed to load gLTF");
    for scene in scenes {
        /*
        println!(
            "Cameras: #{} Lights: #{} Models: #{}",
            scene.cameras.len(),
            scene.lights.len(),
            scene.models.len()
        );
         */
        for model in scene.models {
            match model.mode() {
                Mode::Triangles | Mode::TriangleFan | Mode::TriangleStrip => {
                  let triangles = model.triangles().unwrap();
                  // Render triangles...
                },
                Mode::Lines | Mode::LineLoop | Mode::LineStrip => {
                  let lines = model.lines().unwrap();
                  // Render lines...
                }
                Mode::Points => {
                  let points = model.points().unwrap();
                  // Render points...
                }
              }
        }
    }
}
    
/* 
pub async fn load_model_gltf(
    file_name: &str,
    gpu: &sdl3::gpu::Device,
    queue: &sdl3::gpu::CommandBuffer,
) -> anyhow::Result<bool> {
    let gltf_text  = load_string(file_name).await?;
    let gltf_cursor = Cursor::new(gltf_text);
    let gltf_reader = BufReader::new(gltf_cursor);
    let gltf = Gltf::from_reader(gltf_reader)?;

    //load buffers
    let mut buffer_data = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                // if let Some(blob) = gltf.blob.as_deref() {
                //     buffer_data.push(blob.into());
                //     println!("Found a bin, saving");
                // };
            }
            gltf::buffer::Source::Uri(uri) => {
                let bin = load_binary(uri).await?;
                buffer_data.push(bin);
            }
        }
    }

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            let mesh = node.mesh().expect("Got mesh");
            let primitives = mesh.primitives();
            primitives.for_each(|primitive|) {

                let reader = primitive.reader(|buffer|)
            }
        }
    }

}
*/