//! Applies pooling to the input.
//!
//! This layers looks at adjectant values of the input and then computes a
//! simple pooling operation over them (e.g. taking their maximum or average value).
//! *See [PoolingMode][pooling_mode]*
//!
//! [pooling_mode]: ./enum.PoolingMode.html
//!
//! ## Input Data
//!
//! The layer expects the input to be in either 4D NCHW (2 spatial dimensions)
//! or 5D NCDHW (3 spatial dimensions) format.
use std::rc::Rc;
use co::{IBackend, SharedTensor};
use conn;
use layer::*;
use util::{ArcLock, cast_vec_usize_to_i32};
use super::FilterLayer;

#[derive(Debug, Clone)]
/// [Pooling](./index.html) Layer
pub struct Pooling<T, B: conn::Pooling<T>> {
    mode: PoolingMode,

    filter_shape: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,

    pooling_configs: Vec<Rc<B::CPOOL>>,
}

impl<T, B: conn::Pooling<T>> Pooling<T, B> {
    /// Create a Pooling layer from a PoolingConfig.
    pub fn from_config(config: &PoolingConfig) -> Pooling<T, B> {
        Pooling {
            mode: config.mode,

            filter_shape: config.filter_shape.clone(),
            stride: config.stride.clone(),
            padding: config.padding.clone(),

            pooling_configs: vec![],
        }
    }
}

impl<T, B: conn::Pooling<T>> FilterLayer for Pooling<T, B> {
    /// Calculates the number of spatial dimensions for the pooling operation.
    fn num_spatial_dims(&self, input_shape: &[usize]) -> usize {
        match input_shape.len() {
            4 => 2,
            5 => 3,
            _ => panic!("A pooling layer currently only supports 4D or 5D input.")
        }
    }

    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let num_spatial_dims = self.num_spatial_dims(input_shape);
        let filter = self.spatial_filter_dims(num_spatial_dims);
        let padding = self.padding_dims(num_spatial_dims);
        let stride = self.stride_dims(num_spatial_dims);
        let mut output_shape = Vec::new();
        for dim in &input_shape[0..2].to_vec() {
            output_shape.push(*dim);
        }
        for spatial_dim in Self::calculate_spatial_output_dims(&input_shape[2..], &filter, &padding, &stride) {
            output_shape.push(spatial_dim);
        }

        output_shape
    }

    fn filter_shape(&self) -> &[usize] {
        &self.filter_shape
    }

    fn stride(&self) -> &[usize] {
        &self.stride
    }

    fn padding(&self) -> &[usize] {
        &self.padding
    }
}

impl<B: IBackend + conn::Pooling<f32>> ILayer<B> for Pooling<f32, B> {
    impl_ilayer_common!();

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        for i in 0..input_data.len() {
            let inp = input_data[0].read().unwrap();
            let input_shape = inp.desc();
            let output_shape = self.calculate_output_shape(input_shape);
            output_data[0].write().unwrap().resize(&output_shape).unwrap();
            output_gradient[0].write().unwrap().resize(&output_shape).unwrap();

            let num_spatial_dims = self.num_spatial_dims(inp.desc());
            let filter = cast_vec_usize_to_i32(self.spatial_filter_dims(num_spatial_dims));
            let stride = cast_vec_usize_to_i32(self.stride_dims(num_spatial_dims));
            let padding = cast_vec_usize_to_i32(self.padding_dims(num_spatial_dims));

            let config = backend.new_pooling_config(&filter, &padding, &stride).unwrap();
            self.pooling_configs.push(Rc::new(config));
        }
    }
}

impl<B: IBackend + conn::Pooling<f32>> ComputeOutput<f32, B> for Pooling<f32, B> {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let config = &self.pooling_configs[0];
        match self.mode {
            PoolingMode::Max => backend.pooling_max_plain(input_data[0], output_data[0], &*config).unwrap(),
            // TODO: implement average pooling
            // PoolingMode::Average => unimplemented!(),
        }
    }
}

impl<B: IBackend + conn::Pooling<f32>> ComputeInputGradient<f32, B> for Pooling<f32, B> {
    fn compute_input_gradient(&self,
                              backend: &B,
                              _weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let config = &self.pooling_configs[0];
        match self.mode {
            PoolingMode::Max => backend.pooling_max_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0], config).unwrap()
        }
    }
}

impl<B: IBackend + conn::Pooling<f32>> ComputeParametersGradient<f32, B> for Pooling<f32, B> { }

#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Pooling Layer.
pub struct PoolingConfig {
    /// The PoolingMode to use
    pub mode: PoolingMode,
    /// The shape of the filter
    pub filter_shape: Vec<usize>,
    /// The stride size
    pub stride: Vec<usize>,
    /// The padding size
    pub padding: Vec<usize>,
}

impl Into<LayerType> for PoolingConfig {
    fn into(self) -> LayerType {
        LayerType::Pooling(self)
    }
}

#[derive(Debug, Copy, Clone)]
/// The different modes of pooling that can be calculated.
pub enum PoolingMode {
    /// The maximum value inside the pooling window will be used as result.
    Max,
    // /// The average of all values inside the pooling window will be used as result.
    // Average,
}
