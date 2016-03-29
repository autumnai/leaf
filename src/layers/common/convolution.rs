//! Convolves the input tensor.
//!
//! Computes this convolution with a set of learnable filters,
//! each producing one feature map in the output tensor.
//!
//! [This site][cs231n_convnets] provides a good overview of the functionality
//! of convolutional layers.
//!
//! ## Input Data
//!
//! The layer expects the input to be in 4D NCHW format (2 spatial dimensions).
//!
//! [cs231n_convnets]: https://cs231n.github.io/convolutional-networks
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use co::prelude::*;
use conn;
use conn::ConvolutionConfig as connConvolutionConfig;
use layer::*;
use util::{ArcLock, cast_vec_usize_to_i32};
use weight::FillerType;
use super::FilterLayer;

#[derive(Debug, Clone)]
/// Convolution Layer
pub struct Convolution<B: conn::Convolution<f32>> {
    num_output: usize,
    filter_shape: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,

    workspace: Option<ArcLock<SharedTensor<u8>>>,
    convolution_config: Option<Rc<B::CC>>,
}

impl<B: conn::Convolution<f32>> Convolution<B> {
    /// Create a Convolution layer from a ConvolutionConfig.
    pub fn from_config(config: &ConvolutionConfig) -> Convolution<B> {
        Convolution {
            num_output: config.num_output,

            filter_shape: config.filter_shape.clone(),
            stride: config.stride.clone(),
            padding: config.padding.clone(),

            workspace: None,
            convolution_config: None,
        }
    }

    fn calculate_filter_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let num_spatial_dims = self.num_spatial_dims(input_shape);
        let spatial_dims = self.spatial_filter_dims(num_spatial_dims);
        let filter_n = self.num_output; // number of output feature maps
        let filter_c = input_shape[1]; // number of input feature maps
        let filter_h = spatial_dims[0];
        let filter_w = spatial_dims[1];

        vec![filter_n, filter_c, filter_h, filter_w]
    }

    fn create_filter(&self, device: &DeviceType, input_shape: &[usize]) -> SharedTensor<f32> {
        let filter_shape = self.calculate_filter_shape(input_shape);

        SharedTensor::<f32>::new(device, &filter_shape).unwrap()
    }
}

impl<B: conn::Convolution<f32>> FilterLayer for Convolution<B> {
    /// Calculates the number of spatial dimensions for the convolution operation.
    fn num_spatial_dims(&self, input_shape: &[usize]) -> usize {
        match input_shape.len() {
            4 => 2,
            _ => panic!("Only 2D convolutions supported at the moment")
        }
    }

    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let num_spatial_dims = self.num_spatial_dims(input_shape);
        let filter = self.spatial_filter_dims(num_spatial_dims);
        let padding = self.padding_dims(num_spatial_dims);
        let stride = self.stride_dims(num_spatial_dims);
        let mut output_shape = Vec::new();
        for dim in &input_shape[0..1].to_vec() {
            output_shape.push(*dim);
        }
        output_shape.push(self.num_output);
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

impl<B: IBackend + conn::Convolution<f32>> ILayer<B> for Convolution<B> {
    impl_ilayer_common!();

    fn auto_weight_blobs(&self) -> bool {
        true
    }

    fn reshape(&mut self,
               backend: Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        for i in 0..input_data.len() {
            let inp = input_data[0].read().unwrap();
            let mut output_data = output_data[0].write().unwrap();
            let mut output_gradient = output_gradient[0].write().unwrap();
            let input_shape = inp.desc();
            let output_shape = self.calculate_output_shape(input_shape);
            output_data.resize(&output_shape).unwrap();
            output_gradient.resize(&output_shape).unwrap();

            let device = <B as IBackend>::device(&backend);
            let num_spatial_dims = self.num_spatial_dims(inp.desc());
            let mut filter = self.create_filter(device, input_shape);
            let stride = cast_vec_usize_to_i32(self.stride_dims(num_spatial_dims));
            let padding = cast_vec_usize_to_i32(self.padding_dims(num_spatial_dims));

            let config = backend.new_convolution_config(&inp, &output_data, &mut filter,
                                                        conn::ConvForwardAlgo::Auto, conn::ConvBackwardFilterAlgo::Auto, conn::ConvBackwardDataAlgo::Auto,
                                                        &stride, &padding).unwrap();

            // resize and fill weights
            weights_data[0].write().unwrap().resize(filter.desc()).unwrap();
            let filler = FillerType::Glorot {
                input_size: inp.desc().size(),
                output_size: output_shape.size(),
            };
            filler.fill(&mut weights_data[0].write().unwrap());
            weights_gradient[0].write().unwrap().resize(filter.desc()).unwrap();
            self.convolution_config = Some(Rc::new(config));
        }
    }

    fn resize_shared_workspace(&mut self, backend: Rc<B>, workspace: Option<ArcLock<SharedTensor<u8>>>) -> Option<ArcLock<SharedTensor<u8>>> {
        let required_size = self.convolution_config.as_ref().unwrap().workspace_size();
        let new_workspace = if workspace.is_none() {
            Arc::new(RwLock::new(SharedTensor::<u8>::new(IBackend::device(&*backend), &(required_size)).unwrap()))
        } else {
            let old_workspace = workspace.as_ref().unwrap().clone();
            let old_workspace_size = old_workspace.read().unwrap().capacity();
            if old_workspace_size < required_size {
                Arc::new(RwLock::new(SharedTensor::<u8>::new(IBackend::device(&*backend), &(required_size)).unwrap()))
            } else {
                workspace.unwrap()
            }
        };

        self.workspace = Some(new_workspace.clone());
        Some(new_workspace)
    }
}

impl<B: IBackend + conn::Convolution<f32>> ComputeOutput<f32, B> for Convolution<B> {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let filter_data = weights[0];
        let conv_config = self.convolution_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
        backend.convolution_plain(filter_data, input_data[0], output_data[0], &mut workspace, conv_config).unwrap();
    }
}

impl<B: IBackend + conn::Convolution<f32>> ComputeInputGradient<f32, B> for Convolution<B> {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              _output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let filter_data = weights_data[0];
        let conv_config = self.convolution_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
        // compute gradient w.r.t. input
        backend.convolution_grad_data_plain(filter_data, output_gradients[0], input_gradients[0], &mut workspace, conv_config).unwrap();
    }
}

impl<B: IBackend + conn::Convolution<f32>> ComputeParametersGradient<f32, B> for Convolution<B> {
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   _output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) {
        // TODO: compute gradient w.r.t to bias
        let filter_gradient = &mut parameters_gradients[0];
        let conv_config = self.convolution_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
        // compute gradient w.r.t. filter
        backend.convolution_grad_filter_plain(input_data[0], output_gradients[0], filter_gradient, &mut workspace, conv_config).unwrap();
    }
}


#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Convolution Layer.
pub struct ConvolutionConfig {
    /// The number of output feature maps
    pub num_output: usize,
    /// The size of the kernel
    pub filter_shape: Vec<usize>,
    /// The stride size
    pub stride: Vec<usize>,
    /// The padding size
    pub padding: Vec<usize>,
}

impl Into<LayerType> for ConvolutionConfig {
    fn into(self) -> LayerType {
        LayerType::Convolution(self)
    }
}

#[cfg(test)]
mod tests {
    use co::*;
    use super::{Convolution, ConvolutionConfig};
    use super::super::FilterLayer;

    #[test]
    #[cfg(feature="cuda")]
    fn correct_shapes() {
        let cfg = ConvolutionConfig {
            num_output: 64,

            filter_shape: vec![11],
            padding: vec![2],
            stride: vec![4],
        };
        let layer = Convolution::<Backend<Cuda>>::from_config(&cfg);
        let num_spatial_dims = layer.num_spatial_dims(&[1, 3, 224, 224]);
        assert_eq!(2, num_spatial_dims);
        assert_eq!(vec![11, 11], layer.spatial_filter_dims(2));
        assert_eq!(vec![2, 2], layer.padding_dims(2));
        assert_eq!(vec![4, 4], layer.stride_dims(2));
        assert_eq!(vec![64, 3, 11, 11], layer.calculate_filter_shape(&[1, 3, 224, 224]));
        assert_eq!(vec![1, 64, 55, 55], layer.calculate_output_shape(&[1, 3, 224, 224]));
    }
}
