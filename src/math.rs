use rblas::*;

pub fn leaf_cpu_axpy(alpha: &f32, x: &[f32], y: &mut Vec<f32>) {
    Axpy::axpy(alpha, x, y);
}

pub fn leaf_cpu_axpby(alpha: &f32, x: &[f32], beta: &f32, y: &mut Vec<f32>) {
    leaf_cpu_scal(beta, y);
    leaf_cpu_axpy(alpha, x, y);
}

pub fn leaf_cpu_dot(x: &[f32], y: &[f32]) -> f32 {
    Dot::dot(x, y)
}

pub fn leaf_cpu_scal(alpha: &f32, x: &mut Vec<f32>) {
    Scal::scal(alpha, x)
}
