#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Pos2 {
    pub x: f32,
    pub y: f32,
}

impl std::ops::Add<Pos2> for Pos2 {
    type Output = Pos2;

    fn add(self, other: Pos2) -> Pos2 {
        Pos2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub<Pos2> for Pos2 {
    type Output = Pos2;

    fn sub(self, other: Pos2) -> Pos2 {
        Pos2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32> for Pos2 {
    type Output = Pos2;

    fn mul(self, other: f32) -> Pos2 {
        Pos2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl std::ops::Mul<Pos2> for Pos2 {
    type Output = Pos2;

    fn mul(self, other: Pos2) -> Pos2 {
        Pos2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl std::default::Default for Pos2 {
    fn default() -> Self {
        Pos2 { x: 0.0, y: 0.0 }
    }
}
