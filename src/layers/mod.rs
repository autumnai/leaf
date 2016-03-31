//! Provides the fundamental units of computation in a Neural Network.
//!
//! These layers provide different type of operations to the data Blobs
//! that flow through them.
//! The operations provided by the layers are grouped into five categories:
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
//! * [__Container__][mod_container]</br>
//! Container layers take `LayerConfig`s and connect them on initialization, which
//! creates a "network". But as container layers are layers one can stack multiple
//! container layers on top of another and compose even bigger container layers.
//! Container layers differ in how they connect the layers that it receives.
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
//! [mod_container]: ./container/index.html

/// Implement [ILayer][1] for [activation layers][2].
/// [1]: ./layer/trait.ILayer.html
/// [2]: ./layers/activation/index.html

pub use self::activation::{
    ReLU,
    Sigmoid,
};

#[cfg(all(feature="cuda", not(feature="native")))]
pub use self::common::{
    Convolution, ConvolutionConfig,
    Pooling, PoolingConfig, PoolingMode,
};

pub use self::common::{
    Linear, LinearConfig,
    LogSoftmax,
    Softmax,
};

pub use self::loss::{
    NegativeLogLikelihood, NegativeLogLikelihoodConfig,
};

pub use self::utility::{
    Flatten,
    Reshape, ReshapeConfig,
};

pub use self::container::{
    Sequential, SequentialConfig,
};

pub mod activation;
pub mod common;
pub mod loss;
pub mod utility;
pub mod container;
