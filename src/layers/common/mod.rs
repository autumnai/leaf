//! Provides common neural network layers.
//!
//! For now the layers in common should be discribed as layers that are typical
//! layers for building neural networks but are not activation or loss layers.
pub use self::convolution::Convolution;

pub mod convolution;
