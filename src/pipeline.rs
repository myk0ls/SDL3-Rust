use sdl3::{
    event::Event, gpu::{
        Buffer, BufferBinding, BufferRegion, BufferUsageFlags, ColorTargetDescription, ColorTargetInfo, CompareOp, CopyPass, CullMode, 
        DepthStencilState, DepthStencilTargetInfo, Device, FillMode, Filter, GraphicsPipeline, GraphicsPipelineTargetInfo,
         IndexElementSize, LoadOp, PrimitiveType, RasterizerState, SampleCount, SamplerAddressMode, SamplerCreateInfo, 
         SamplerMipmapMode, ShaderFormat, ShaderStage, StoreOp, Texture, TextureCreateInfo, TextureFormat, TextureRegion, 
         TextureSamplerBinding, TextureTransferInfo, TextureType, TextureUsage, TransferBuffer, TransferBufferLocation, 
         TransferBufferUsage, VertexAttribute, VertexBufferDescription, VertexElementFormat, VertexInputRate, VertexInputState,
    }, keyboard::Keycode, pixels::Color, surface::Surface, video::Window, Error
};

use std::mem::size_of;

use crate::vertex;

pub fn create_pipeline(gpu: &Device, window: &Window) -> Result<GraphicsPipeline, Error> {
        // Our shaders, require to be precompiled by a SPIR-V compiler beforehand
        let vert_shader = gpu
        .create_shader()
        .with_code(
            ShaderFormat::SpirV,
            include_bytes!("../shaders/cube-texture.vert.spv"),
            ShaderStage::Vertex,
        )
        .with_uniform_buffers(3)
        .with_entrypoint("main")
        .build()?;
    let frag_shader = gpu
        .create_shader()
        .with_code(
            ShaderFormat::SpirV,
            include_bytes!("../shaders/cube-texture.frag.spv"),
            ShaderStage::Fragment,
        )
        .with_samplers(1)
        .with_entrypoint("main")
        .build()?;
    
    // Create a pipeline, we specify that we want our target format in the swapchain
    // since we are rendering directly to the screen. However, we could specify a texture
    // buffer instead (e.g., for offscreen rendering).
    let swapchain_format = gpu.get_swapchain_texture_format(&window);
    let pipeline = gpu
        .create_graphics_pipeline()
        .with_primitive_type(PrimitiveType::TriangleList)
        .with_fragment_shader(&frag_shader)
        .with_vertex_shader(&vert_shader)
        .with_vertex_input_state(
            VertexInputState::new()
                .with_vertex_buffer_descriptions(&[VertexBufferDescription::new()
                    .with_slot(0)
                    .with_pitch(size_of::<vertex::Vertex>() as u32)
                    .with_input_rate(VertexInputRate::Vertex)
                    .with_instance_step_rate(0)])
                .with_vertex_attributes(&[
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float3)
                        .with_location(0)
                        .with_buffer_slot(0)
                        .with_offset(0),
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float2)
                        .with_location(1)
                        .with_buffer_slot(0)
                        .with_offset((3 * size_of::<f32>()) as u32),
                ]),
        )
        .with_rasterizer_state(
            RasterizerState::new()
                .with_fill_mode(FillMode::Fill)
                // Turn off culling so that I don't have to get my cube vertex order perfect
                .with_cull_mode(CullMode::None),
        )
        .with_depth_stencil_state(
            // Enable depth testing
            DepthStencilState::new()
                .with_enable_depth_test(true)
                .with_enable_depth_write(true)
                .with_compare_op(CompareOp::Less),
        )
        .with_target_info(
            GraphicsPipelineTargetInfo::new()
                .with_color_target_descriptions(&[
                    ColorTargetDescription::new().with_format(swapchain_format)
                ])
                .with_has_depth_stencil_target(true)
                .with_depth_stencil_format(TextureFormat::D16Unorm),
        )
        .build()?;

    // The pipeline now holds copies of our shaders, so we can release them
    drop(vert_shader);
    drop(frag_shader);

    Ok(pipeline)
}