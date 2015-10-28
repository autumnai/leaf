use shared_memory::*;
use layer::*;

pub struct SigmoidLayer;

impl ILayer for SigmoidLayer {
    impl_neuron_layer!();

    fn forward_cpu(&self, bottom: &[HeapBlob], top: &mut Vec<HeapBlob>) {
        let bottom_data = bottom[0].cpu_data();
        let top_data = top[0].mutable_cpu_data();

        for (i, _) in bottom_data.iter().enumerate() {
            top_data[i] = SigmoidLayer::sigmoid(bottom_data[i])
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
                bottom_diff[i] = top_diff[i] * SigmoidLayer::sigmoid_prime_precalc(sigmoid_x)
            }
        }
    }
}

impl SigmoidLayer {
    fn sigmoid(z: f32) -> f32 {
        1f32 / (1f32 + (-z).exp())
    }

    fn sigmoid_prime(z: f32) -> f32 {
        SigmoidLayer::sigmoid_prime_precalc(SigmoidLayer::sigmoid(z))
    }

    fn sigmoid_prime_precalc(sigmoid_z: f32) -> f32 {
        sigmoid_z * (1f32 - sigmoid_z)
    }
}
