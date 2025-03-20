extern crate sdl3;

use sdl3::{
    event::Event,
    gpu::{
        ColorTargetDescription, ColorTargetInfo, Device, FillMode, GraphicsPipelineTargetInfo,
        LoadOp, PrimitiveType, ShaderFormat, ShaderStage, StoreOp,
    },
    keyboard::Keycode,
    pixels::Color,
};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sld_context = sdl3::init().unwrap();
    let video_subsystem = sld_context.video().unwrap();
    let window = video_subsystem
        .window("rust-sdl3 demo gpu triangle", 800, 600)
        .position_centered()
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    let gpu = Device::new(
        ShaderFormat::SpirV | ShaderFormat::Dxil | ShaderFormat::Dxbc | ShaderFormat::MetalLib, 
        true,
    )?
    .with_window(&window)?;

    let fs_source = include_bytes!(".././shaders/triangle.frag.spv");
    let vs_source = include_bytes!(".././shaders/triangle.vert.spv");

    let vs_shader = gpu
        .create_shader()
        .with_code(ShaderFormat::SpirV, vs_source, ShaderStage::Vertex)
        .with_entrypoint("main")
        .build()?;

    let fs_shader = gpu
        .create_shader()
        .with_code(ShaderFormat::SpirV, fs_source, ShaderStage::Fragment)
        .with_entrypoint("main")
        .build()?;

    let swapchain_format = gpu.get_swapchain_texture_format(&window);

    let pipeline = gpu
        .create_graphics_pipeline()
        .with_fragment_shader(&fs_shader)
        .with_vertex_shader(&vs_shader)
        .with_primitive_type(PrimitiveType::TriangleList)
        .with_fill_mode(FillMode::Fill)
        .with_target_info(GraphicsPipelineTargetInfo::new().with_color_target_descriptions(&[ColorTargetDescription::new().with_format(swapchain_format)])).build()?;

        drop(vs_shader);
        drop(fs_shader);

        let mut event_pump = sld_context.event_pump()?;
        println!(
            "This example demonstrates that the gpu is working"
        );
        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => break 'running,
                    _ => {}
                }
            }

            let mut command_buffer = gpu.acquire_command_buffer()?;
            if let Ok(swapchain) = command_buffer.wait_and_acquire_swapchain_texture(&window) {
                let color_targets = [
                    ColorTargetInfo::default()
                        .with_texture(&swapchain)
                        .with_load_op(LoadOp::Clear)
                        .with_store_op(StoreOp::Store)
                        .with_clear_color(Color::RGB(5, 3, 255)),
                ];
                let render_pass = gpu.begin_render_pass(&command_buffer, &color_targets, None)?;
                render_pass.bind_graphics_pipeline(&pipeline);
                render_pass.draw_primitives(3, 1, 0, 0);
                gpu.end_render_pass(render_pass);
                command_buffer.submit()?;
            } else {
                command_buffer.cancel();
            }
        }

    Ok(())
}
