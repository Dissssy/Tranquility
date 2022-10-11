use std::ops::{Add, Mul, Sub};

#[derive(Debug)]
pub struct Smooth<T> {
    pub value: T,
    pub velocity: T,
    pub target: T,
    pub speed: f32,
    pub damping: f32,
}

impl<T> Smooth<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<f32, Output = T>
        + std::default::Default
        + Mul<T, Output = T>,
{
    pub fn new(value: T) -> Self {
        Self {
            value,
            velocity: T::default(),
            target: value,
            speed: 0.01,
            damping: 0.9,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // let delta = self.target - self.value;
        // self.velocity = (self.velocity * self.damping) + (delta * self.speed);
        // self.value = self.value + self.velocity;

        // update velocity and then value based on dt
        let delta = self.target - self.value;
        self.velocity = (self.velocity * self.damping) + (delta * self.speed);
        self.value = self.value + (self.velocity * dt);
    }
}
