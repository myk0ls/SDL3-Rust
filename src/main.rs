mod camera;
mod pipeline;
mod vertex;
mod build;
mod resources;

use camera::Camera;
use easy_gltf::model::Mode;
use gltf::{material, texture::{self}};
use image::GenericImageView;
use pipeline::create_pipeline;
use vertex::Vertex;
use sdl3::{
    event::Event,
    gpu::{
        Buffer, BufferBinding, BufferRegion, BufferUsageFlags, ColorTargetDescription, ColorTargetInfo, CompareOp, CopyPass, CullMode, DepthStencilState, DepthStencilTargetInfo, Device, FillMode, Filter, GraphicsPipelineTargetInfo, IndexElementSize, LoadOp, PrimitiveType, RasterizerState, SampleCount, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderFormat, ShaderStage, StoreOp, Texture, TextureCreateInfo, TextureFormat, TextureRegion, TextureSamplerBinding, TextureTransferInfo, TextureType, TextureUsage, TransferBuffer, TransferBufferLocation, TransferBufferUsage, VertexAttribute, VertexBufferDescription, VertexElementFormat, VertexInputRate, VertexInputState
    },
    keyboard::{Keycode, Scancode},
    pixels::Color,
    surface::Surface,
    Error,
};
use std::{num, path::Path, ptr::null, sync::Arc};
use ultraviolet::Vec3;

extern crate sdl3;

struct ModelMaterial {
    base_color_texture: Option<Texture<'static>>,
    base_color_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    normal_texture: Option<Texture<'static>>,
    occlusion_texture: Option<Texture<'static>>,
    emissive_texture: Option<Texture<'static>>,
    emissive_factor: [f32; 3],
    texture_sampler: Sampler,
}

impl ModelMaterial {
    fn from_gltf(
        gpu: &Device,
        material: &easy_gltf::Material,
        copy_pass: &CopyPass,
    ) -> Result<Self, Error> {
        // Create sampler (you can customize these parameters)
        let texture_sampler = gpu.create_sampler(
            SamplerCreateInfo::new()
                .with_min_filter(Filter::Linear)
                .with_mag_filter(Filter::Linear)
                .with_mipmap_mode(SamplerMipmapMode::Linear)
                .with_address_mode_u(SamplerAddressMode::Repeat)
                .with_address_mode_v(SamplerAddressMode::Repeat)
                .with_address_mode_w(SamplerAddressMode::Repeat),
        )?;

        // Load base color texture if available
        let base_color_texture = if let Some(texture) = &material.pbr.base_color_texture {
            let image_data = texture.as_raw();
            Some(create_texture_from_gltf(
                gpu,
                image_data,
                texture.width(),
                texture.height(),
                copy_pass,
            )?)
        } else {
            None
        };

        // Similarly load other textures (normal, occlusion, emissive) if needed

        Ok(Self {
            base_color_texture,
            base_color_factor: material.pbr.base_color_factor.into(),
            metallic_factor: material.pbr.metallic_factor,
            roughness_factor: material.pbr.roughness_factor,
            normal_texture: None, // Implement similar to base color
            occlusion_texture: None, // Implement similar to base color
            emissive_texture: None, // Implement similar to base color
            emissive_factor: [0.0, 0.0, 0.0],
            texture_sampler,
        })
    }
}

const WINDOW_HEIGHT: u32 = 900;
const WINDOW_WIDTH: u32 = 1600;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("rust-sdl3 demo: GPU (texture)", WINDOW_WIDTH, WINDOW_HEIGHT)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    let gpu = sdl3::gpu::Device::new(
        ShaderFormat::SpirV | ShaderFormat::Dxil | ShaderFormat::Dxbc | ShaderFormat::MetalLib,
        true,
    )?
    .with_window(&window)?;

    let pipeline = create_pipeline(&gpu, &window)?;

    // We need to start a copy pass in order to transfer data to the GPU
    let copy_commands = gpu.acquire_command_buffer()?;
    let copy_pass = gpu.begin_copy_pass(&copy_commands)?;

    let scenes = easy_gltf::load("./assets/Monkey.glb").expect("Failed to load gLTF");

    // Create containers to store extracted data
    let mut collected_vertices = Vec::new();
    let mut collected_indices = Vec::new();
    //let mut material = Arc::new(easy_gltf::Material { pbr: (), normal: (), occlusion: (), emissive: () });
    let mut materials = Vec::new();

    for scene in scenes {
        for model in scene.models {
            // Convert vertices slice to owned Vec
            let vertices = model.vertices().to_vec();
            // Convert indices Option<&[u32]> to Option<Vec<u32>>
            let indices = model.indices().map(|i| i.to_vec());

            //let material = model.material();

            match model.mode() {
                Mode::Triangles => {
                    collected_vertices.extend(vertices);
                    if let Some(model_indices) = indices {
                        collected_indices.extend(model_indices);
                    }
                },
                // Handle other modes if needed
                _ => {}
            }
            

            let pbr = &model.material().pbr;
            println!("Base color factor: {:?}", pbr.base_color_factor);
            println!("Metallic factor: {}", pbr.metallic_factor);
            println!("Roughness factor: {}", pbr.roughness_factor);
            
            // Access texture information if available
            if let Some(base_color_texture) = &pbr.base_color_texture {
                println!("Base color texture: {:?}", base_color_texture.to_ascii_lowercase());
            }

            // Create material for this model
            let material = ModelMaterial::from_gltf(&gpu, &*model.material(), &copy_pass)?;
            materials.push(material);
            
        }
    }

    // Now you can use collected_vertices and collected_indices for buffer creation
    let vertices_len_bytes = collected_vertices.len() * std::mem::size_of::<easy_gltf::model::Vertex>();
    let indices_len_bytes = collected_indices.len() * std::mem::size_of::<u32>();

    // Create transfer buffer using the actual loaded data sizes
    let transfer_buffer = gpu
        .create_transfer_buffer()
        .with_size(vertices_len_bytes.max(indices_len_bytes) as u32)
        .with_usage(TransferBufferUsage::Upload)
        .build()?;

    /*
    // Next, we create a transfer buffer that is large enough to hold either
    // our vertices or indices since we will be transferring both with it.
    let vertices_len_bytes = CUBE_VERTICES.len() * size_of::<Vertex>();
    let indices_len_bytes = CUBE_INDICES.len() * size_of::<u16>();
    let transfer_buffer = gpu
        .create_transfer_buffer()
        .with_size(vertices_len_bytes.max(indices_len_bytes) as u32)
        .with_usage(TransferBufferUsage::Upload)
        .build()?;
    */

    // Create GPU buffers to hold our vertices and indices and transfer data to them
    let vertex_buffer = create_buffer_with_data(
        &gpu,
        &transfer_buffer,
        &copy_pass,
        BufferUsageFlags::Vertex,
        &collected_vertices,
    )?;
    let index_buffer = create_buffer_with_data(
        &gpu,
        &transfer_buffer,
        &copy_pass,
        BufferUsageFlags::Index,
        &collected_indices,
    )?;


    // We're done with the transfer buffer now, so release it.
    drop(transfer_buffer);

    // Load up a texture to put on the cube
    let cube_texture = create_texture_from_image(&gpu, "./assets/texture.bmp", &copy_pass)?;

    // And configure a sampler for pulling pixels from that texture in the frag shader
    let cube_texture_sampler = gpu.create_sampler(
        SamplerCreateInfo::new()
            .with_min_filter(Filter::Nearest)
            .with_mag_filter(Filter::Nearest)
            .with_mipmap_mode(SamplerMipmapMode::Nearest)
            .with_address_mode_u(SamplerAddressMode::Repeat)
            .with_address_mode_v(SamplerAddressMode::Repeat)
            .with_address_mode_w(SamplerAddressMode::Repeat),
    )?;

    let default_texture = create_default_texture(&gpu, &copy_pass)?;

    // Now complete and submit the copy pass commands to actually do the transfer work
    gpu.end_copy_pass(copy_pass);
    copy_commands.submit()?;

    // We'll need to allocate a texture buffer for our depth buffer for depth testing to work
    let mut depth_texture = gpu.create_texture(
        TextureCreateInfo::new()
            .with_type(TextureType::_2D)
            .with_width(WINDOW_WIDTH)
            .with_height(WINDOW_HEIGHT)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
            .with_sample_count(SampleCount::NoMultiSampling)
            .with_format(TextureFormat::D16Unorm)
            .with_usage(TextureUsage::Sampler | TextureUsage::DepthStencilTarget),
    )?;

    //create the camera
    let mut camera = Camera::new(65.0, WINDOW_WIDTH, WINDOW_HEIGHT, 0.1, 100.0);

    //hide cursor, capture mouse and restrict to window
    sdl_context.mouse().set_relative_mouse_mode(&window, true);
    sdl_context.mouse().show_cursor(false);

    //resources::load_gltf("./assets/Monkey.glb");


    let mut state: [bool; 6] = [false; 6];

    let mut rotation = 45.0f32;
    let mut event_pump = sdl_context.event_pump()?;
    'running: loop {
        
        state[0] = event_pump.keyboard_state().is_scancode_pressed(Scancode::W);
        state[1] = event_pump.keyboard_state().is_scancode_pressed(Scancode::A);
        state[2] = event_pump.keyboard_state().is_scancode_pressed(Scancode::S);
        state[3] = event_pump.keyboard_state().is_scancode_pressed(Scancode::D);
        state[4] = event_pump.keyboard_state().is_scancode_pressed(Scancode::Space);
        state[5] = event_pump.keyboard_state().is_scancode_pressed(Scancode::LCtrl);


        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::F8),
                    ..
                } => break 'running,
                Event::KeyDown { keycode: Some(key), ..} => {
                    match key {
                        //Keycode::Space => camera.move_camera(Vec3::new(0.0, move_speed, 0.0)),
                        //Keycode::LShift => camera.move_camera(Vec3::new(0.0, -move_speed, 0.0)),
                        _ => {}
                    }
                }
                Event::MouseMotion { xrel, yrel, .. } => {
                    // Handle camera rotation
                    let sensitivity = 0.005;
                    camera.rotate_camera(-yrel as f32 * sensitivity, xrel as f32 * sensitivity);
                }
                _ => {}
                
            }
        }

        let move_speed = 0.03;
        //W
        if state[0] {
            camera.move_camera(Vec3::new(0.0, 0.0, move_speed));
        }

        //A
        if state[1] {
            camera.move_camera(Vec3::new(-move_speed, 0.0, 0.0));
        }

        //S
        if state[2] {
            camera.move_camera(Vec3::new(0.0, 0.0, -move_speed));
        }

        //D
        if state[3] {
            camera.move_camera(Vec3::new(move_speed, 0.0, 0.0));
        }

        //Space
        if state[4] {
            camera.move_camera(Vec3::new(0.0, move_speed, 0.0));
        }

        //LCtrl
        if state[5] {
            camera.move_camera(Vec3::new(0.0, -move_speed, 0.0));
        }
        
        
        
        //Update camera view_matrix
        camera.update_view_matrix();
        //let debug_camera_position = &camera.position();

        //println!("View Matrix: {:?}", camera.view_matrix());
        //println!("Projection Matrix: {:?}", camera.projection_matrix());


        //println!("{x} {y} {z}", x = debug_camera_position.x.to_string(), y = debug_camera_position.y.to_string(), z = debug_camera_position.z.to_string());

        // The swapchain texture is basically the framebuffer corresponding to the drawable
        // area of a given window - note how we "wait" for it to come up
        //
        // This is because a swapchain needs to be "allocated", and it can quickly run out
        // if we don't properly time the rendering process.
        let mut command_buffer = gpu.acquire_command_buffer()?;
        if let Ok(swapchain) = command_buffer.wait_and_acquire_swapchain_texture(&window) {
            // Again, like in gpu-clear.rs, we'd want to define basic operations for our cube
            let color_targets = [ColorTargetInfo::default()
                .with_texture(&swapchain)
                .with_load_op(LoadOp::Clear)
                .with_store_op(StoreOp::Store)
                .with_clear_color(Color::RGB(128, 128, 128))];
            // This time, however, we want depth testing, so we need to also target a depth texture buffer
            let depth_target = DepthStencilTargetInfo::new()
                .with_texture(&mut depth_texture)
                .with_cycle(true)
                .with_clear_depth(1.0)
                .with_clear_stencil(0)
                .with_load_op(LoadOp::Clear)
                .with_store_op(StoreOp::Store)
                .with_stencil_load_op(LoadOp::Clear)
                .with_stencil_store_op(StoreOp::Store);
            let render_pass =
                gpu.begin_render_pass(&command_buffer, &color_targets, Some(&depth_target))?;

            // Screen is cleared below due to the color target info
            render_pass.bind_graphics_pipeline(&pipeline);

            // Now we'll bind our buffers/sampler and draw the cube
            render_pass.bind_vertex_buffers(
                0,
                &[BufferBinding::new()
                    .with_buffer(&vertex_buffer)
                    .with_offset(0)],
            );
            render_pass.bind_index_buffer(
                &BufferBinding::new()
                    .with_buffer(&index_buffer)
                    .with_offset(0),
                IndexElementSize::_32Bit,
            );

            /* 
            render_pass.bind_fragment_samplers(
                0,
                &[TextureSamplerBinding::new()
                    .with_texture(&cube_texture)
                    .with_sampler(&cube_texture_sampler)],
            );
            */
            

            for material in &materials {
                let texture = &material.base_color_texture.as_ref().unwrap_or(&default_texture);
                    render_pass.bind_fragment_samplers(
                        0, 
                    &[TextureSamplerBinding::new()
                        .with_texture(texture)
                        .with_sampler(&material.texture_sampler)]
                    );
                

                command_buffer.push_fragment_uniform_data(0, &material.base_color_factor);
                //command_buffer.push_fragment_uniform_data(1, &material.metallic_factor);
                //command_buffer.push_fragment_uniform_data(2, &material.roughness_factor);

                
            }

            // Set the rotation uniform for our cube vert shader
            command_buffer.push_vertex_uniform_data(0, &rotation);
            rotation += 0.1f32;

            let view_matrix_data: [f32; 16] = mat4_to_array(*camera.view_matrix());
            let projection_matrix_data: [f32; 16] = mat4_to_array(*camera.projection_matrix());

            command_buffer.push_vertex_uniform_data(1, &view_matrix_data);
            command_buffer.push_vertex_uniform_data(2, &projection_matrix_data);

            //println!("View Matrix Data: {:?}", view_matrix_data);
            //println!("Projection Matrix Data: {:?}", projection_matrix_data);

            // Finally, draw the cube
            render_pass.draw_indexed_primitives(collected_indices.len() as u32, 1, 0, 0, 0);

            gpu.end_render_pass(render_pass);
            command_buffer.submit()?;
        } else {
            // Swapchain unavailable, cancel work
            command_buffer.cancel();
        }
    }

    Ok(())
}

fn create_texture_from_image(
    gpu: &Device,
    image_path: impl AsRef<Path>,
    copy_pass: &CopyPass,
) -> Result<Texture<'static>, Error> {
    let image = Surface::load_bmp(image_path.as_ref())?;
    let image_size = image.size();
    let size_bytes =
        image.pixel_format().byte_size_per_pixel() as u32 * image_size.0 * image_size.1;

    let texture = gpu.create_texture(
        TextureCreateInfo::new()
            .with_format(TextureFormat::R8g8b8a8Unorm)
            .with_type(TextureType::_2D)
            .with_width(image_size.0)
            .with_height(image_size.1)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
            .with_usage(TextureUsage::Sampler),
    )?;

    let transfer_buffer = gpu
        .create_transfer_buffer()
        .with_size(size_bytes)
        .with_usage(TransferBufferUsage::Upload)
        .build()?;

    let mut buffer_mem = transfer_buffer.map::<u8>(gpu, false);
    image.with_lock(|image_bytes| {
        buffer_mem.mem_mut().copy_from_slice(image_bytes);
    });
    buffer_mem.unmap();

    copy_pass.upload_to_gpu_texture(
        TextureTransferInfo::new()
            .with_transfer_buffer(&transfer_buffer)
            .with_offset(0),
        TextureRegion::new()
            .with_texture(&texture)
            .with_layer(0)
            .with_width(image_size.0)
            .with_height(image_size.1)
            .with_depth(1),
        false,
    );

    Ok(texture)
}

/* 
fn create_texture_from_gltf(
    gpu: &Device,
    texture: &easy_gltf::Material::Texture,
    copy_pass: &CopyPass,
) -> Result<(Texture<'static>, Sampler), Error> {
    
} 
*/

fn create_texture_from_gltf(
    gpu: &Device,
    image_data: &[u8],
    width: u32,
    height: u32,
    copy_pass: &CopyPass,
) -> Result<Texture<'static>, Error> {
    let size_bytes = width * height * 4; // Assuming RGBA8 format
    
    let texture = gpu.create_texture(
        TextureCreateInfo::new()
            .with_format(TextureFormat::R8g8b8a8Unorm)
            .with_type(TextureType::_2D)
            .with_width(width)
            .with_height(height)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
            .with_usage(TextureUsage::Sampler),
    )?;

    let transfer_buffer = gpu
        .create_transfer_buffer()
        .with_size(size_bytes)
        .with_usage(TransferBufferUsage::Upload)
        .build()?;

    let mut buffer_mem = transfer_buffer.map::<u8>(gpu, false);
    buffer_mem.mem_mut().copy_from_slice(&image_data);
    buffer_mem.unmap();

    copy_pass.upload_to_gpu_texture(
        TextureTransferInfo::new()
            .with_transfer_buffer(&transfer_buffer)
            .with_offset(0),
        TextureRegion::new()
            .with_texture(&texture)
            .with_layer(0)
            .with_width(width)
            .with_height(height)
            .with_depth(1),
        false,
    );

    Ok(texture)
}

/// Creates a GPU buffer and uploads data to it using the given `copy_pass` and `transfer_buffer`.
fn create_buffer_with_data<T: Copy>(
    gpu: &Device,
    transfer_buffer: &TransferBuffer,
    copy_pass: &CopyPass,
    usage: BufferUsageFlags,
    data: &[T],
) -> Result<Buffer, Error> {
    // Figure out the length of the data in bytes
    let len_bytes = data.len() * std::mem::size_of::<T>();

    // Create the buffer with the size and usage we want
    let buffer = gpu
        .create_buffer()
        .with_size(len_bytes as u32)
        .with_usage(usage)
        .build()?;

    // Map the transfer buffer's memory into a place we can copy into, and copy the data
    //
    // Note: We set `cycle` to true since we're reusing the same transfer buffer to
    // initialize both the vertex and index buffer. This makes SDL synchronize the transfers
    // so that one doesn't interfere with the other.
    let mut map = transfer_buffer.map::<T>(gpu, true);
    let mem = map.mem_mut();
    for (index, &value) in data.iter().enumerate() {
        mem[index] = value;
    }

    // Now unmap the memory since we're done copying
    map.unmap();
    // Finally, add a command to the copy pass to upload this data to the GPU
    //
    // Note: We also set `cycle` to true here for the same reason.
    copy_pass.upload_to_gpu_buffer(
        TransferBufferLocation::new()
            .with_offset(0)
            .with_transfer_buffer(transfer_buffer),
        BufferRegion::new()
            .with_offset(0)
            .with_size(len_bytes as u32)
            .with_buffer(&buffer),
        true,
    );

    Ok(buffer)
}

fn create_default_texture(gpu: &Device, copy_pass: &CopyPass) -> Result<Texture<'static>, Error> {
    // 1x1 white pixel
    let white_pixel = [255, 255, 255, 255];
    let texture = gpu.create_texture(
        TextureCreateInfo::new()
            .with_format(TextureFormat::R8g8b8a8Unorm)
            .with_type(TextureType::_2D)
            .with_width(1)
            .with_height(1)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
            .with_usage(TextureUsage::Sampler),
    )?;
    
    let transfer_buffer = gpu.create_transfer_buffer()
        .with_size(4)
        .build()?;
    
    let mut buffer_mem = transfer_buffer.map::<u8>(gpu, false);
    buffer_mem.mem_mut().copy_from_slice(&white_pixel);
    buffer_mem.unmap();

    copy_pass.upload_to_gpu_texture(
        TextureTransferInfo::new()
            .with_transfer_buffer(&transfer_buffer)
            .with_offset(0),
        TextureRegion::new()
            .with_texture(&texture)
            .with_layer(0)
            .with_width(1)
            .with_height(1)
            .with_depth(1),
        false,
    );

    Ok(texture)
}

fn mat4_to_array(mat: ultraviolet::Mat4) -> [f32; 16] {
    let mut array = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            array[i * 4 + j] = mat.cols[i][j];
        }
    }
    array
}