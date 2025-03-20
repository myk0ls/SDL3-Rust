pub struct Camera {
    position: [f32; 3],      // (x, y, z)
    front: [f32; 3],         // Direction the camera is looking
    up: [f32; 3],            // Up vector
    right: [f32; 3],         // Right vector
    world_up: [f32; 3],      // World's up vector (usually [0.0, 1.0, 0.0])
    yaw: f32,                // Horizontal rotation
    pitch: f32,              // Vertical rotation
    speed: f32,              // Movement speed
    sensitivity: f32,        // Mouse sensitivity
}

impl Camera {
    pub fn new(position: [f32; 3], yaw: f32, pitch: f32)) -> Self {
        let world_up = [0.0, 1.0, 0.0];
        let front = [0.0, 0.0, -1.0];
        let right = [0.0, 0.0, 0.0]; // Will be calculated later
        let up = [0.0, 0.0, 0.0];    // Will be calculated later

        let mut camera = Camera {
            position,
            front,
            up,
            right,
            world_up,
            yaw,
            pitch,
            speed: 2.5,
            sensitivity: 0.1,
        };

        camera.update_vectors();
        camera
    }

    pub fn update(&mut self, mouse_delta: (f32, f32)) {
        // Update yaw and pitch based on mouse movement
        self.yaw += mouse_delta.0 * self.sensitivity;
        self.pitch -= mouse_delta.1 * self.sensitivity;

        // Clamp pitch to avoid flipping
        self.pitch = self.pitch.clamp(-89.0, 89.0);

        // Update front, right, and up vectors
        self.update_vectors();
    }

    fn update_vectors(&mut self) {
        // Calculate the new front vector
        let front_x = self.yaw.to_radians().cos() * self.pitch.to_radians().cos();
        let front_y = self.pitch.to_radians().sin();
        let front_z = self.yaw.to_radians().sin() * self.pitch.to_radians().cos();
        self.front = [front_x, front_y, front_z];

        // Recalculate right and up vectors
        self.right = normalize(cross(self.front, self.world_up));
        self.up = normalize(cross(self.right, self.front));
    }

    pub fn get_view_matrix(&self) -> [[f32; 4]; 4] {
        let target = [
            self.position[0] + self.front[0],
            self.position[1] + self.front[1],
            self.position[2] + self.front[2],
        ];

        // Create a look-at matrix
        look_at(self.position, target, self.up)
    }
}