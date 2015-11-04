use rblas::Axpy;
use rblas::Dot;

pub fn leaf_cpu_dot(x: &[f32], y: &[f32]) -> f32 {
    Dot::dot(x, y)
}

pub fn leaf_cpu_axpy(alpha: &f32, x: &[f32], y: &mut Vec<f32>) {
    Axpy::axpy(alpha, x, y);
}
