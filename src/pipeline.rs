use easy_gltf::model::Vertex;
use sdl3::{
        gpu::{
        ColorTargetDescription, CompareOp, CullMode, 
        DepthStencilState,  Device, FillMode,  GraphicsPipeline, GraphicsPipelineTargetInfo,
         PrimitiveType, RasterizerState,
          ShaderFormat, ShaderStage, TextureFormat,
         VertexAttribute, VertexBufferDescription, VertexElementFormat, VertexInputRate, VertexInputState,
    }, video::Window, Error
};

use std::{collections::HashMap, mem::size_of};

use crate::vertex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineType {
    Sky,
    Shadow,
    Opaque,
    Ssao,
    Composite,
    Transparent,
    Raycast,
    Ui,
    Random,
}

pub struct RenderPipeline {
    pub pipeline: GraphicsPipeline,
}

pub struct PipelineManager {
    gpu: Device,
    pipelines: HashMap<PipelineType, RenderPipeline>,
    swapchain_format: TextureFormat,
}

impl PipelineManager {
    pub fn new(gpu: Device, window: &Window) -> Result<Self, Error> {
        let swapchain_format = gpu.get_swapchain_texture_format(&window);
        Ok(PipelineManager { gpu, pipelines: HashMap::new(), swapchain_format })
    }

    pub fn load_pipelines(&mut self) -> Result<(), Error> {
        let opaque_pipeline = self.create_opaque_pipeline()?;
        self.pipelines.insert(PipelineType::Opaque, opaque_pipeline);
        let shadow_pipeline = self.create_shadow_pipeline()?;
        self.pipelines.insert(PipelineType::Shadow, shadow_pipeline);
        Ok(())
    }

    pub fn get_pipeline(&self, pipeline_type: PipelineType) -> Result<&GraphicsPipeline, Box<dyn std::error::Error>> {
        self.pipelines
            .get(&pipeline_type)
            .map(|rp| &rp.pipeline)
            .ok_or(format!("Pipeline {:?} not found", pipeline_type).into())
    }

   
    fn create_shadow_pipeline(&self) -> Result<RenderPipeline, Error> {
        let vert_shader = self.gpu
            .create_shader()
            .with_code(
                ShaderFormat::SpirV,
                include_bytes!("../shaders/shadow.vert.spv"),
                ShaderStage::Vertex,
            )
            .with_uniform_buffers(1)
            .with_entrypoint("main")
            .build()?;

        let frag_shader = self.gpu
            .create_shader()
            .with_code(
                ShaderFormat::SpirV, 
                include_bytes!("../shaders/shadow.frag.spv"), 
                ShaderStage::Fragment,
            )
            .with_entrypoint("main")
            .build()?;

        let pipeline = self.gpu
        .create_graphics_pipeline()
        .with_primitive_type(PrimitiveType::TriangleList)
        .with_vertex_shader(&vert_shader)
        .with_fragment_shader(&frag_shader)
        .with_vertex_input_state(
            VertexInputState::new()
                .with_vertex_buffer_descriptions(&[VertexBufferDescription::new()
                    .with_slot(0)
                    .with_pitch(std::mem::size_of::<easy_gltf::model::Vertex>() as u32)
                    .with_input_rate(VertexInputRate::Vertex)
                    .with_instance_step_rate(0)])
                .with_vertex_attributes(&[
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float3)
                        .with_location(0)
                        .with_buffer_slot(0)
                        .with_offset(0),
                ]),
        )
        .with_rasterizer_state(
            RasterizerState::new()
                .with_fill_mode(FillMode::Fill)
                .with_cull_mode(CullMode::Back) // Cull back faces for shadows
                .with_depth_bias_constant_factor(2.0), // Add depth bias to combat shadow acne
                //.with_depth_bias_slope_scale(2.0),
        )
        .with_depth_stencil_state(
            DepthStencilState::new()
                .with_enable_depth_test(true)
                .with_enable_depth_write(true)
                .with_compare_op(CompareOp::LessOrEqual),
        )
        .with_target_info(
            GraphicsPipelineTargetInfo::new()
                .with_has_depth_stencil_target(true)
                .with_depth_stencil_format(TextureFormat::D32Float), // Higher precision for shadows
        )
        .build()?;

        drop(vert_shader);
        drop(frag_shader);

        Ok(RenderPipeline { pipeline })
    }

    fn create_opaque_pipeline(&self) -> Result<RenderPipeline, Error> {
            // Our shaders, require to be precompiled by a SPIR-V compiler beforehand
        let vert_shader = self.gpu
            .create_shader()
            .with_code(
                ShaderFormat::SpirV,
                include_bytes!("../shaders/cube-texture.vert.spv"),
                ShaderStage::Vertex,
            )
            .with_uniform_buffers(3)
            .with_entrypoint("main")
            .build()?;
        let frag_shader = self.gpu
            .create_shader()
            .with_code(
                ShaderFormat::SpirV,
                include_bytes!("../shaders/cube-texture.frag.spv"),
                ShaderStage::Fragment,
            )
            .with_samplers(2)
            .with_entrypoint("main")
            .with_uniform_buffers(2)
            .build()?;
        
        // Create a pipeline, we specify that we want our target format in the swapchain
        // since we are rendering directly to the screen. However, we could specify a texture
        // buffer instead (e.g., for offscreen rendering).
        //let swapchain_format = self.gpu.get_swapchain_texture_format(self.swapchain_format);
        let pipeline = self.gpu
            .create_graphics_pipeline()
            .with_primitive_type(PrimitiveType::TriangleList)
            .with_fragment_shader(&frag_shader)
            .with_vertex_shader(&vert_shader)
            .with_vertex_input_state(
                VertexInputState::new()
                    .with_vertex_buffer_descriptions(&[VertexBufferDescription::new()
                        .with_slot(0)
                        .with_pitch(std::mem::size_of::<easy_gltf::model::Vertex>() as u32) // Should be 3+3+4+2 = 12 floats = 48 bytes
                        .with_input_rate(VertexInputRate::Vertex)
                        .with_instance_step_rate(0)])
                    .with_vertex_attributes(&[
                        // Position (vec3) - 0-12 bytes
                        VertexAttribute::new()
                            .with_format(VertexElementFormat::Float3)
                            .with_location(0)
                            .with_buffer_slot(0)
                            .with_offset(0),
                        
                        // Normal (vec3) - 12-24 bytes
                        VertexAttribute::new()
                            .with_format(VertexElementFormat::Float3)
                            .with_location(1)
                            .with_buffer_slot(0)
                            .with_offset(12),
                        
                        // Tangent (vec4) - 24-40 bytes
                        VertexAttribute::new()
                            .with_format(VertexElementFormat::Float4)
                            .with_location(2)
                            .with_buffer_slot(0)
                            .with_offset(24),
                        
                        // Texture Coord (vec2) - 40-48 bytes
                        VertexAttribute::new()
                            .with_format(VertexElementFormat::Float2)
                            .with_location(3)
                            .with_buffer_slot(0)
                            .with_offset(40),
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
                        ColorTargetDescription::new().with_format(self.swapchain_format)
                    ])
                    .with_has_depth_stencil_target(true)
                    .with_depth_stencil_format(TextureFormat::D16Unorm),
            )
            .build()?;

        // The pipeline now holds copies of our shaders, so we can release them
        drop(vert_shader);
        drop(frag_shader);

        Ok(RenderPipeline { pipeline })
    }
}


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
        .with_uniform_buffers(1)
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
                    .with_pitch(std::mem::size_of::<easy_gltf::model::Vertex>() as u32) // Should be 3+3+4+2 = 12 floats = 48 bytes
                    .with_input_rate(VertexInputRate::Vertex)
                    .with_instance_step_rate(0)])
                .with_vertex_attributes(&[
                    // Position (vec3) - 0-12 bytes
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float3)
                        .with_location(0)
                        .with_buffer_slot(0)
                        .with_offset(0),
                    
                    // Normal (vec3) - 12-24 bytes
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float3)
                        .with_location(1)
                        .with_buffer_slot(0)
                        .with_offset(12),
                    
                    // Tangent (vec4) - 24-40 bytes
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float4)
                        .with_location(2)
                        .with_buffer_slot(0)
                        .with_offset(24),
                    
                    // Texture Coord (vec2) - 40-48 bytes
                    VertexAttribute::new()
                        .with_format(VertexElementFormat::Float2)
                        .with_location(3)
                        .with_buffer_slot(0)
                        .with_offset(40),
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