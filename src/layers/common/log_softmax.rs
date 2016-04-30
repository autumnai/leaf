//! Computes the logarithmic softmax of its input.
//!
use co::{IBackend, SharedTensor};
use conn;
use layer::*;
use util::ArcLock;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// LogSoftmax Layer
pub struct LogSoftmax;

impl<B: IBackend + conn::LogSoftmax<f32>> ILayer<B> for LogSoftmax {
    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        let input_desc = input_data[0].read().unwrap().desc().clone();
        input_gradient[0].write().unwrap().resize(&input_desc).unwrap();
        output_data[0].write().unwrap().resize(&input_desc).unwrap();
        output_gradient[0].write().unwrap().resize(&input_desc).unwrap();
    }
}

impl<B: IBackend + conn::LogSoftmax<f32>> ComputeOutput<f32, B> for LogSoftmax {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        backend.log_softmax(input_data[0], output_data[0]).unwrap();
    }
}

impl<B: IBackend + conn::LogSoftmax<f32>> ComputeInputGradient<f32, B> for LogSoftmax {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        backend.log_softmax_grad(output_data[0], output_gradients[0],
                                 input_gradients[0]).unwrap();
    }
}

impl<B: IBackend + conn::LogSoftmax<f32>> ComputeParametersGradient<f32, B> for LogSoftmax { }

impl ::std::default::Default for LogSoftmax {
    fn default() -> LogSoftmax {
        LogSoftmax
    }
}
