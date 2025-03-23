use ultraviolet::projection::rh_ydown::perspective_vk;
use::ultraviolet::Vec3;
use::ultraviolet::Mat4;

pub struct Camera {
    position: Vec3,
    pitch: f32,
    yaw: f32,
    fov: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
    dirty: bool, //mark if camera needs updating
    view_matrix: Mat4,
    projection_matrix: Mat4,

}

impl Camera {
    pub fn new(fov: f32, width: u32, height: u32, near: f32, far: f32) -> Self {
        let aspect_ratio = width as f32 / height as f32;
        let projection_matrix = perspective_vk(fov.to_radians(), aspect_ratio, near, far);
        let view_matrix = Mat4::identity();

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

        let rotation_x = Mat4::from_rotation_x(self.pitch);
        let rotation_y = Mat4::from_rotation_y(self.yaw);
        let rotation = rotation_y * rotation_x;

        let translation = Mat4::from_translation(-self.position);
        self.view_matrix = rotation * translation;
        self.dirty = false;
    }

    pub fn move_camera(&mut self, delta: Vec3) {
        self.position += delta;
        self.dirty = true;
    }

    pub fn rotate_camera(&mut self, delta_pitch: f32, delta_yaw: f32) {
        self.pitch += delta_pitch;
        self.yaw += delta_yaw;
        self.dirty = true;
    }

    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
        self.projection_matrix = perspective_vk(self.fov.to_radians(), self.aspect_ratio, self.near, self.far);
        self.dirty = true;
    }

}