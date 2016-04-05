//! Provides nonlinear activation methods.
//!
//! Activation Layers take a input tensor, provide the activation operation and
//! produce a output tensor.
//! Thanks to the nonlinearity of the activation methods, we can 'learn' and
//! detect nonlinearities
//! in our (complex) datasets.
//!
//! The activation operation used should depend on the task at hand. For binary
//! classification a
//! step function might be very useful. For more complex tasks continious
//! activation functions such
//! as [Sigmoid][mod_sigmoid], TanH, [ReLU][mod_relu] should be used. In most cases ReLU might
//! provide the best results.
//!
//! If you supply the same blob as input and output to a layer via the [LayerConfig][struct_layerconfig],
//! computations will be done in-place, requiring less memory.
//!
//! The activation function is also sometimes called transfer function.
//!
//! [mod_sigmoid]: ./sigmoid/index.html
//! [mod_relu]: ./relu/index.html
//! [struct_layerconfig]: ../../layer/struct.LayerConfig.html
#[macro_export]
macro_rules! impl_ilayer_activation {
    () => (
        fn exact_num_output_blobs(&self) -> Option<usize> { Some(1) }
        fn exact_num_input_blobs(&self) -> Option<usize> { Some(1) }
    )
}

pub use self::relu::ReLU;
pub use self::sigmoid::Sigmoid;
pub use self::tanh::TanH;

pub mod relu;
pub mod sigmoid;
pub mod tanh;
