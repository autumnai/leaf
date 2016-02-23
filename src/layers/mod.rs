//! Provides the fundamental units of computation for the [Network][1].
//! [1]: ../network/index.html
//!
//! These layers provide different type of operations to the data Blobs
//! that flow through them.
//! The operations provided by the layers can be
//! roughly grouped into four categories:
//!
//! * [__Activation__][mod_activation]</br>
//! Activation Layers provide element-wise operations and produce one top Blob
//! of the same size as the bottom Blob.
//! It can be seen as a synonym to nonlinear [Activation Functions][2].
//!
//! * [__Common__][mod_common]</br>
//! Common Layers can differ in their connectivity and behavior and are
//! typically all network layer
//! types which are not covered by activation or loss layers.
//! Examples would be fully connected
//! layers, covolutional layers, pooling layers, etc.
//!
//! * [__Loss__][mod_loss]</br>
//! Loss Layers compare an output to a target value and assign cost to
//! minimize. Loss Layers are often the last layer in a [Network][1].
//!
//! * [__Utility__][mod_utility]</br>
//! Utility Layers provide all kind of helpful functionality, which might not
//! be directly related
//! to machine learning and neural nets. This could be operations for
//! normalizing,
//! restructuring or transforming information, log and debug behavior or data
//! access.
//! Utility Layers follow the general behavior of a layer, like the other types
//! do.
//!
//! For more information about how these layers work together, see the
//! documentation for the general [Layer module][3].
//!
//! [2]: https://en.wikipedia.org/wiki/Activation_function
//! [3]: ../layer/index.html
//!
//! [mod_activation]: ./activation/index.html
//! [mod_common]: ./common/index.html
//! [mod_loss]: ./loss/index.html
//! [mod_utility]: ./utility/index.html

/// Implement [ILayer][1] for [activation layers][2].
/// [1]: ./layer/trait.ILayer.html
/// [2]: ./layers/activation/index.html

#[allow(unused_import_braces)]
pub use self::activation::{
    ReLU,
    Sigmoid,
};

#[allow(unused_import_braces)]
pub use self::common::{
    Convolution, ConvolutionConfig,
    Linear, LinearConfig,
    LogSoftmax,
    Pooling, PoolingConfig, PoolingMode,
    Softmax,
};

#[allow(unused_import_braces)]
pub use self::loss::{
    NegativeLogLikelihood, NegativeLogLikelihoodConfig,
};

#[allow(unused_import_braces)]
pub use self::utility::{
    Flatten,
    Reshape, ReshapeConfig,
};

pub mod activation;
pub mod common;
pub mod loss;
pub mod utility;
