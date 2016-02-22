//! Provides common neural network layers.
//!
//! For now the layers in common should be discribed as layers that are typical
//! layers for building neural networks but are not activation or loss layers.
#[macro_export]
macro_rules! impl_ilayer_common {
    () => (
        fn exact_num_output_blobs(&self) -> Option<usize> { Some(1) }
        fn exact_num_input_blobs(&self) -> Option<usize> { Some(1) }
    )
}

pub use self::convolution::{Convolution, ConvolutionConfig};
pub use self::linear::{Linear, LinearConfig};
pub use self::log_softmax::LogSoftmax;
pub use self::pooling::{Pooling, PoolingConfig, PoolingMode};
pub use self::softmax::Softmax;

pub mod convolution;
pub mod linear;
pub mod log_softmax;
pub mod pooling;
pub mod softmax;

/// Provides common utilities for Layers that utilize a filter with stride and padding.
///
/// This is used by the Convolution and Pooling layers.
pub trait FilterLayer {
    /// Computes the shape of the spatial dimensions.
    fn calculate_spatial_output_dims(input_dims: &[usize], filter_dims: &[usize], padding: &[usize], stride: &[usize]) -> Vec<usize> {
        let mut output_dims = Vec::with_capacity(input_dims.len());
        for (i, _) in input_dims.iter().enumerate() {
            output_dims.push(((input_dims[i] + (2 * padding[i]) - filter_dims[i]) / stride[i]) + 1);
        }
        output_dims
    }

    /// Calculate output shape based on the shape of filter, padding, stride and input.
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize>;

    /// Calculates the number of spatial dimensions for the pooling operation.
    fn num_spatial_dims(&self, input_shape: &[usize]) -> usize;

    /// Retrievs the spatial dimensions for the filter based on `self.filter_shape()`
    /// and the number of spatial dimensions.
    ///
    /// The spatial dimensions only make up part of the whole filter shape. The other parts are the
    /// number of input and output feature maps.
    fn spatial_filter_dims(&self, num_spatial_dims: usize) -> Vec<usize> {
        let mut spatial_dims = Vec::with_capacity(num_spatial_dims);
        let filter_shape = self.filter_shape();
        if filter_shape.len() == 1 {
            for i in 0..num_spatial_dims {
                spatial_dims.push(filter_shape[0]);
            }
        } else if filter_shape.len() == num_spatial_dims {
            panic!("unimplemented: You can not yet specify one filter dimension per spatial dimension");
        } else {
            panic!("Must either specify one filter_shape or one filter_shape per spatial dimension. Supplied {:?}", filter_shape.len());
        }

        spatial_dims
    }

    /// Retrievs the stride for the convolution based on `self.stride`
    /// and the number of spatial dimensions.
    fn stride_dims(&self, num_spatial_dims: usize) -> Vec<usize> {
        let mut stride_dims = Vec::with_capacity(num_spatial_dims);
        let stride = self.stride();
        if stride.len() == 1 {
            for i in 0..num_spatial_dims {
                stride_dims.push(stride[0]);
            }
        } else if stride.len() == num_spatial_dims {
            panic!("unimplemented: You can not yet specify one stride per spatial dimension");
        } else {
            panic!("Must either specify one stride or one stride per spatial dimension. Supplied {:?}", stride.len());
        }

        stride_dims
    }

    /// Retrievs the padding for the convolution based on `self.padding`
    /// and the number of spatial dimensions.
    fn padding_dims(&self, num_spatial_dims: usize) -> Vec<usize> {
        let mut padding_dims = Vec::with_capacity(num_spatial_dims);
        let padding = self.padding();
        if padding.len() == 1 {
            for i in 0..num_spatial_dims {
                padding_dims.push(padding[0]);
            }
        } else if padding.len() == num_spatial_dims {
            panic!("unimplemented: You can not yet specify one padding per spatial dimension");
        } else {
            panic!("Must either specify one padding or one padding per spatial dimension. Supplied {:?}", padding.len());
        }

        padding_dims
    }

    /// The filter_shape that will be used by `spatial_filter_dims`.
    fn filter_shape(&self) -> &[usize];

    /// The stride that will be used by `stride_dims`.
    fn stride(&self) -> &[usize];

    /// The padding that will be used by `padding_dims`.
    fn padding(&self) -> &[usize];
}
