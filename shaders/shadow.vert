#version 450

layout (location = 0) in vec3 pos;

layout(set = 0, binding = 0) uniform LightViewProjection {
    mat4 light_view_proj;
};

void main() {
    gl_Position = light_view_proj * vec4(pos, 1.0);
}