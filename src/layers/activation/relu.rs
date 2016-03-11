//! Applies the nonlinear Rectified Linear Unit.
//!
//! Non-linearity activation function: y = max(0, x)
//!
//! This is generally the preferred choice over Sigmod or TanH.
//! The max function used in ReLU is usually faster to compute than the exponentiation
//! needed in a Sigmoid layer.

use co::{IBackend,SharedTensor};
use conn::Relu;
#[cfg(all(feature="cuda", not(feature="native")))]
use conn::ReluPointwise;
use layer::*;
use util::ArcLock;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// ReLU Activation Layer
pub struct ReLU;

//
// ReLU + ReLUPointwise
// Only on CUDA
//
#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + Relu<f32> + ReluPointwise<f32>> ILayer<B> for ReLU {
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

#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + Relu<f32> + ReluPointwise<f32>> ComputeOutput<f32, B> for ReLU {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        match input_data.get(0) {
            Some(input) => backend.relu_plain(input, output_data[0]).unwrap(),
            None => backend.relu_pointwise_plain(output_data[0]).unwrap(),
        }
    }
}

#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + Relu<f32> + ReluPointwise<f32>> ComputeInputGradient<f32, B> for ReLU {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        match output_data.get(0) {
            Some(_) => backend.relu_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => backend.relu_pointwise_grad_plain(input_data[0], input_gradients[0]).unwrap(),
        }
    }
}

#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + Relu<f32> + ReluPointwise<f32>> ComputeParametersGradient<f32, B> for ReLU {}

//
// ReLU without ReLUPointwise
// Only on CUDA
//
#[cfg(feature="native")]
impl<B: IBackend + Relu<f32>> ILayer<B> for ReLU {
    impl_ilayer_activation!();

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

#[cfg(feature="native")]
impl<B: IBackend + Relu<f32>> ComputeOutput<f32, B> for ReLU {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        match input_data.get(0) {
            Some(input) => backend.relu_plain(input, output_data[0]).unwrap(),
            None => panic!("No input provided for ReLU layer."),
        }
    }
}

#[cfg(feature="native")]
impl<B: IBackend + Relu<f32>> ComputeInputGradient<f32, B> for ReLU {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        match output_data.get(0) {
            Some(_) => backend.relu_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => panic!("No output_data provided for ReLU layer backward."),
        }
    }
}

#[cfg(feature="native")]
impl<B: IBackend + Relu<f32>> ComputeParametersGradient<f32, B> for ReLU {}
