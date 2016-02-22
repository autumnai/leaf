//! Applies a linear transformation to the input data `y = a * x + b`
//!
//! The variables are:
//!
//! - `y`: output value
//! - `a`: weight (a trainable weight in a neural network)
//! - `x`: input value
//! - `b`: bias (not implemented yet)
//!
//! ## Input
//!
//! The input can either have one or two dimensions:
//!
//! - If the input has one dimension the transformation will just be applied to the input data.
//! - If the input has two dimensions **the first dimension is treated as batch size** (`N`)
//!   and the transformation will be applied to every vector in the second dimension, using the
//!   same weights and biases.
//!
//! In the context of convolutional neural networks this layer is also
//! called a "fully-connected layer" if it is used at the end of the network.
use std::rc::Rc;
use co::backend::IBackend;
use co::tensor::SharedTensor;
use coblas::transpose::Transpose;
use coblas::plugin::*;
use layer::*;
use util::{ArcLock, native_scalar, LayerOps};
use weight::FillerType;

#[derive(Debug)]
/// Linear Layer
pub struct Linear {
    output_size: usize,

    one: SharedTensor<f32>,
    zero: SharedTensor<f32>,
}

impl Linear {
    /// Create a Linear layer from a LinearConfig.
    pub fn from_config(config: &LinearConfig) -> Linear {
        let one = native_scalar(1f32);
        let zero = native_scalar(0f32);

        Linear {
            output_size: config.output_size,

            one: one,
            zero: zero,
        }
    }

    // Calculates the input size by skipping the batch size.
    fn calculate_input_size(input_shape: &[usize]) -> usize {
        input_shape.iter().skip(1).fold(1, |prod, i| prod * i)
    }

    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let n = input_shape[0]; // batch size
        vec![n, self.output_size]
    }

    fn calculate_weight_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let m = Self::calculate_input_size(input_shape);
        vec![self.output_size, m]
    }
}

impl<B: IBackend + LayerOps<f32>> ILayer<B> for Linear {
    impl_ilayer_common!();

    fn init(&mut self, backend: Rc<B>) {
        let device = <B as IBackend>::device(&backend);
        let _ = self.one.add_device(device);
        self.one.sync(device).unwrap();
        let _ = self.zero.add_device(device);
        self.zero.sync(device).unwrap();
    }

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        let input = input_data[0].read().unwrap();
        // reshape top
        let output_shape = self.calculate_output_shape(input.desc());
        output_data[0].write().unwrap().resize(&output_shape).unwrap();
        output_gradient[0].write().unwrap().resize(&output_shape).unwrap();
        // reshape weight
        let weight_shape = self.calculate_weight_shape(input.desc());
        // TODO: change weight creation to not require this
        if let Some(weight) = weights_data.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
            let filler = FillerType::Glorot {
                input_size: Self::calculate_input_size(input.desc()),
                output_size: self.output_size,
            };
            filler.fill(&mut weight.write().unwrap());

            let native_backend = ::util::native_backend();
            let bound_weight = weight.read().unwrap();
            let native_output = bound_weight.get(native_backend.device()).unwrap().as_native().unwrap();
        }
        if let Some(weight) = weights_gradient.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
        }
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeOutput<f32, B> for Linear {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        backend.gemm_plain(&self.one, Transpose::NoTrans, input_data[0], Transpose::Trans, weights[0], &self.zero, output_data[0]).unwrap();
        let has_bias_term = false; // TODO: implement bias term
        if has_bias_term {
            let bias_multiplier = unimplemented!();
            let bias_data = unimplemented!();
            backend.gemm_plain(&self.one, Transpose::NoTrans, bias_multiplier, Transpose::NoTrans, bias_data, &self.one, output_data[0]).unwrap();
        }
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeInputGradient<f32, B> for Linear {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        // Gradient with respect to input data
        backend.gemm_plain(&self.one, Transpose::NoTrans, output_gradients[0], Transpose::NoTrans, weights_data[0], &self.zero, input_gradients[0]).unwrap();
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeParametersGradient<f32, B> for Linear {
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) {
        // gradient w.r.t. weights
        backend.gemm_plain(&self.one, Transpose::Trans, output_gradients[0], Transpose::NoTrans, input_data[0], &self.zero, parameters_gradients[0]).unwrap();

        // TODO: implement gradient w.r.t bias
        // if (bias_term_ && this->param_propagate_down_[1]) {
        //     const Dtype* top_diff = top[0]->gpu_diff();
        //     // Gradient with respect to bias
        //     caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        //         bias_multiplier_.gpu_data(), (Dtype)1.,
        //         this->blobs_[1]->mutable_gpu_diff());
        // }
    }
}

impl ::std::default::Default for Linear {
    fn default() -> Linear {
        let config = LinearConfig {
            output_size: 10,
        };

        Self::from_config(&config)
    }
}


#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Specifies configuration parameters for a Linear Layer.
pub struct LinearConfig {
    /// The number of output values
    pub output_size: usize,
}
