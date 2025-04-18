#version 450

// Texture sampler
layout (set = 2, binding = 0) uniform sampler2D tex_sampler;

// Shadow map
layout (set = 2, binding = 1) uniform sampler2DShadow shadow_map;

//Light matrix
layout (set = 3, binding = 1) uniform LightMatrix {
    mat4 light_view_proj;
};

// Texture coordinates from the vertex shader
layout (location = 0) in vec2 tex_coord;

// Color from our vertex shader
layout (location = 1) in vec3 frag_color;

layout (location = 2) in vec3 pos;

// Final color of the pixel
layout (location = 0) out vec4 final_color;

layout(set = 3, binding = 0) uniform PushConstants {
	vec4 base_color_factor;
} material;

float calculate_shadow(vec3 object_pos) {
    vec4 light_space_pos = light_view_proj * vec4(object_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    //final_color = vec4(proj_coords, 1.0);
    return texture(shadow_map, proj_coords);
}


void main() {
	vec4 base_color = texture(tex_sampler, tex_coord) * material.base_color_factor;
    mat4 model_matrix = mat4(1.0);

    vec3 world_pos = (model_matrix * vec4(pos, 1.0)).xyz;

    
    float shadow = calculate_shadow(world_pos);
    
	//final_color = base_color * (1.0 - shadow * 0.5);
    //final_color = vec4(shadow, shadow, shadow, 1.0);
    //final_color = vec4(texture(shadow_map, vec3(0.5, 0.5, 0.5)));
    final_color = vec4(vec3(texture(shadow_map, vec3(0.5, 0.5, 0.3))), 1.0);
}