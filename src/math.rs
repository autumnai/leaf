use rblas::Dot;

pub fn leaf_cpu_dot(x: &[f32], y: &[f32]) -> f32 {
    // return Dot::dot(x, y[..x.len()]);
    Dot::dot(x, y)
}
