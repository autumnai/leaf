//! Provides the fundamental units of computation for the [Network][1].
//! [1]: ../network/index.html
//!
//! These layers provide different type of operations to the data Blobs that flow through them.  
//! The operations provided by the layers can be grouped roughly into four categories:
//!
//! * __Activation__  
//! Activation Layers provide element-wise operations and produce one top Blob of the same size as
//! the bottom Blob. It can be seen as a synonym to nonlinear [Activation Functions][2].
//!
//! * __Common__  
//! Common Layers can differ in their connectivity and behavior and are typically all network layer
//! types which are not covered by activation or loss layers. Examples would be fully connected
//! layers, covolutional layers, pooling layers, etc.
//!
//! * __Loss__  
//! Loss Layers compare an output to a target value and assign cost to minimize. Loss Layers are
//! often the last layer in a [Network][1].
//!
//! * __Utility__  
//! Utility Layers provide all kind of helpful functionality, which might not be directly related
//! to machine learning and neural nets. This could be operations for normalizing,
//! restructuring or transforming information, log and debug behavior or data access.
//! Utility Layers follow the general behavior of a layer, like the other types do.
//!
//! For more information about how all these layers work specifically, see the documentation for
//! the general [Layer module][3].
//!
//! ## Examples
//!
//! ```
//! extern crate leaf;
//! use leaf::layers::*;
//!
//! # fn main() {
//! let _ = activation::Sigmoid;
//! let _ = common::Convolution;
//! let _ = loss::Softmax;
//! let _ = utility::Flatten;
//! # }
//! ```
//!
//! [2]: https://en.wikipedia.org/wiki/Activation_function
//! [3]: ../layer/index.html
macro_rules! impl_neuron_layer {
    () => (
        fn exact_num_top_blobs(&self) -> usize { 1 }
        fn exact_num_bottom_blobs(&self) -> usize { 1 }
    )
}

#[allow(unused_import_braces)]
pub use self::activation::{Sigmoid};

#[allow(unused_import_braces)]
pub use self::common::{Convolution};

#[allow(unused_import_braces)]
pub use self::loss::{Softmax};

#[allow(unused_import_braces)]
pub use self::utility::{Flatten};

pub mod activation;
pub mod common;
pub mod loss;
pub mod utility;
