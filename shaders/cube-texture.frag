#version 450

// Texture sampler
layout (set = 2, binding = 0) uniform sampler2D tex_sampler;

// Texture coordinates from the vertex shader
layout (location = 0) in vec2 tex_coord;

// Color from our vertex shader
layout (location = 1) in vec3 frag_color;

// Final color of the pixel
layout (location = 0) out vec4 final_color;

// Push constants instead of uniform block
//layout(push_constant) uniform Material {
    //vec4 base_color_factor;
    //float metallic_factor;
    //float roughness_factor;
//} material;

layout(set = 3, binding = 0) uniform PushConstants {
	vec4 base_color_factor;
} material;


void main() {
	vec4 base_color = texture(tex_sampler, tex_coord) * material.base_color_factor;

	final_color = base_color;
}