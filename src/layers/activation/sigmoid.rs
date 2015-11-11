//! Applies the nonlinear Log-Sigmoid function.
//!
//! Non-linearity activation function: y = (1 + e^(-x))^(-1)
//!
//! A classic choice in neural networks.
//! But you might consider using ReLu as an alternative.
//!
//! ReLu, compared to Sigmoid
//!
//! * reduces the likelyhood of vanishing gradients
//! * increases the likelyhood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this
//! writing (2015).
use shared_memory::*;
use layer::*;

#[derive(Debug, Copy, Clone)]
/// Sigmoid Activation Layer
pub struct Sigmoid;

impl ILayer for Sigmoid {
    impl_ilayer_activation!();

    fn forward_cpu(&self, bottom: &[ReadBlob], top: &mut Vec<&mut WriteBlob>) {
        let bottom_data = bottom[0].cpu_data();
        let top_data = top[0].mutable_cpu_data();

        for (i, _) in bottom_data.iter().enumerate() {
            top_data[i] = Sigmoid::sigmoid(bottom_data[i])
        }
    }

    fn backward_cpu(&self, top: &[HeapBlob], propagate_down: &[bool], bottom: &mut Vec<HeapBlob>) {
        if propagate_down[0] {
            let top_data = top[0].cpu_data();
            let top_diff = top[0].cpu_diff();
            let count = bottom[0].len();
            let bottom_diff = bottom[0].mutable_cpu_diff();

            for i in 0..count {
                let sigmoid_x = top_data[i];
                // bottom_diff[i] = top_diff[i] * sigmoid_x * (1f32 - sigmoid_x);
                bottom_diff[i] = top_diff[i] * Sigmoid::sigmoid_prime_precalc(sigmoid_x)
            }
        }
    }
}

impl Sigmoid {
    fn sigmoid(z: f32) -> f32 {
        1f32 / (1f32 + (-z).exp())
    }

    fn sigmoid_prime(z: f32) -> f32 {
        Sigmoid::sigmoid_prime_precalc(Sigmoid::sigmoid(z))
    }

    fn sigmoid_prime_precalc(sigmoid_z: f32) -> f32 {
        sigmoid_z * (1f32 - sigmoid_z)
    }
}
