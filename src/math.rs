use rblas::Dot;

pub fn leaf_cpu_dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    // return Dot::dot(x, y[..x.len()]);
    return Dot::dot(x, y);
}
