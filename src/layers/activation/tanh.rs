//! Applies the nonlinear TanH function.
//!
//! Non-linearity activation function: y = sinh(x) / cosh(x)
//!
//! You might consider using ReLU as an alternative.
//!
//! ReLU, compared to TanH:
//!
//! * reduces the likelyhood of vanishing gradients
//! * increases the likelyhood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this writing (2016).
use co::{IBackend, SharedTensor};
use conn;
use layer::*;
use util::ArcLock;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// TanH Activation Layer
pub struct TanH;

//
// Tanh + TanhPointwise
// Only on CUDA
#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + conn::Tanh<f32> + conn::TanhPointwise<f32>> ILayer<B> for TanH {
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
impl<B: IBackend + conn::Tanh<f32> + conn::TanhPointwise<f32>> ComputeOutput<f32, B> for TanH {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        match input_data.get(0) {
            Some(input) => backend.tanh_plain(input, output_data[0]).unwrap(),
            None => backend.tanh_pointwise_plain(output_data[0]).unwrap(),
        }
    }
}

#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + conn::Tanh<f32> + conn::TanhPointwise<f32>> ComputeInputGradient<f32, B> for TanH {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        match output_data.get(0) {
            Some(_) => backend.tanh_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => backend.tanh_pointwise_grad_plain(input_data[0], input_gradients[0]).unwrap(),
        }
    }
}

#[cfg(all(feature="cuda", not(feature="native")))]
impl<B: IBackend + conn::Tanh<f32> + conn::TanhPointwise<f32>> ComputeParametersGradient<f32, B> for TanH {}

//
// Tanh without TanhPointwise
// Only on CUDA
//
#[cfg(feature="native")]
impl<B: IBackend + conn::Tanh<f32>> ILayer<B> for TanH {
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
impl<B: IBackend + conn::Tanh<f32>> ComputeOutput<f32, B> for TanH {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        match input_data.get(0) {
            Some(input) => backend.tanh_plain(input, output_data[0]).unwrap(),
            None => panic!("No input provided for TanH layer."),
        }
    }
}

#[cfg(feature="native")]
impl<B: IBackend + conn::Tanh<f32>> ComputeInputGradient<f32, B> for TanH {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        match output_data.get(0) {
            Some(_) => backend.tanh_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => panic!("No output_data provided for TanH layer backward."),
        }
    }
}

#[cfg(feature="native")]
impl<B: IBackend + conn::Tanh<f32>> ComputeParametersGradient<f32, B> for TanH {}
