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
use co::{IBackend, SharedTensor};
use conn;
use layer::*;
use util::ArcLock;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Sigmoid Activation Layer
pub struct Sigmoid;

impl<B: IBackend + conn::Sigmoid<f32> + conn::SigmoidPointwise<f32>> ILayer<B> for Sigmoid {
    impl_ilayer_activation!();

    fn compute_in_place(&self) -> bool {
        true
    }

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        if let Some(inp) = input_data.get(0) {
            let read_inp = inp.read().unwrap();
            let input_desc = read_inp.desc();
            input_gradient[0].write().unwrap().resize(input_desc).unwrap();
            output_data[0].write().unwrap().resize(input_desc).unwrap();
            output_gradient[0].write().unwrap().resize(input_desc).unwrap();
        }
    }
}

impl<B: IBackend + conn::Sigmoid<f32> + conn::SigmoidPointwise<f32>> ComputeOutput<f32, B> for Sigmoid {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        match input_data.get(0) {
            Some(input) => backend.sigmoid_plain(input, output_data[0]).unwrap(),
            None => backend.sigmoid_pointwise_plain(output_data[0]).unwrap(),
        }
    }
}

impl<B: IBackend + conn::Sigmoid<f32> + conn::SigmoidPointwise<f32>> ComputeInputGradient<f32, B> for Sigmoid {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        match output_data.get(0) {
            Some(_) => backend.sigmoid_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => backend.sigmoid_pointwise_grad_plain(input_data[0], input_gradients[0]).unwrap(),
        }
    }
}

impl<B: IBackend + conn::Sigmoid<f32> + conn::SigmoidPointwise<f32>> ComputeParametersGradient<f32, B> for Sigmoid {}
