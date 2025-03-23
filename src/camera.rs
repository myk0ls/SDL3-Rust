use ultraviolet::projection::rh_ydown::perspective_vk;
use ultraviolet::{Vec3, Mat4};

pub struct Camera {
    position: Vec3,
    pitch: f32,
    yaw: f32,
    fov: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
    dirty: bool,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

impl Camera {
    pub fn new(fov: f32, width: u32, height: u32, near: f32, far: f32) -> Self {
        let aspect_ratio = width as f32 / height as f32;
        let projection_matrix = perspective_vk(fov.to_radians(), aspect_ratio, near, far);
        let view_matrix = Mat4::identity();

        // Initialize camera vectors
        let forward = Vec3::new(0.0, 0.0, -1.0); // Default forward direction (negative Z)
        let right = Vec3::new(1.0, 0.0, 0.0);    // Default right direction (positive X)
        let up = Vec3::new(0.0, 1.0, 0.0);       // Default up direction (positive Y)

        Self {
            position: Vec3::zero(),
            pitch: 0.0,
            yaw: 0.0,
            fov,
            aspect_ratio,
            near,
            far,
            dirty: true,
            view_matrix,
            projection_matrix,
            forward,
            right,
            up,
        }
    }

    pub fn position(&self) -> &Vec3 {
        &self.position
    }

    pub fn projection_matrix(&self) -> &Mat4 {
        &self.projection_matrix
    }

    pub fn view_matrix(&self) -> &Mat4 {
        &self.view_matrix
    }

    pub fn update_view_matrix(&mut self) {
        if !self.dirty {
            return;
        }

        // Calculate the view matrix using the camera's position and orientation
        let target = self.position + self.forward;
        let world_up = Vec3::new(0.0, 1.0, 0.0); // Fixed world up direction
        self.view_matrix = Mat4::look_at(self.position, target, world_up);

        self.dirty = false;
    }

    pub fn move_camera(&mut self, delta: Vec3) {
        // Move the camera relative to its orientation
        self.position += self.forward * delta.z; // Forward/backward
        self.position += self.right * delta.x;   // Left/right
        self.position += self.up * delta.y;      // Up/down

        self.dirty = true;
    }

    pub fn rotate_camera(&mut self, delta_pitch: f32, delta_yaw: f32) {
        // Update pitch and yaw
        self.pitch += delta_pitch;
        self.yaw += delta_yaw;

        // Restrict pitch to avoid flipping the camera
        self.pitch = self.pitch.clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());

        // Calculate the new forward vector
        self.forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalized();

        // Recalculate the right vector using a fixed world up direction
        let world_up = Vec3::new(0.0, 1.0, 0.0); // Fixed world up direction
        self.right = self.forward.cross(world_up).normalized();

        // Recalculate the up vector using the right and forward vectors
        self.up = self.right.cross(self.forward).normalized();

        self.dirty = true;
    }

    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
        self.projection_matrix = perspective_vk(self.fov.to_radians(), self.aspect_ratio, self.near, self.far);
        self.dirty = true;
    }
}