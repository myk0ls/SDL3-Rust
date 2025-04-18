#version 450

// layout (location = 0) in vec3 pos;

// layout(set = 1, binding = 0) uniform LightViewProjection {
//     mat4 light_view_proj;
// };

// void main() {
//     gl_Position = light_view_proj * vec4(pos, 1.0);
// }

layout (location = 0) in vec3 pos;
layout(set = 1, binding = 0) uniform LightViewProjection {
    mat4 light_view_proj;
};
layout (location = 1) out float depth_value;

void main() {
    vec4 clip_pos = light_view_proj * vec4(pos, 1.0);
    gl_Position = clip_pos;
    depth_value = clip_pos.z / clip_pos.w; // depth
}