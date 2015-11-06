//! Provides nonlinear activation methods.
//!
//! Activation Layers take a bottom Blob, provide the activation operation and produce a top Blob.
//! Thanks to the nonlinearity of the activation methods, we can 'learn' and detect nonlinearities
//! in our (complex) datasets.
//!
//! The activation operation used should depend on the task at hand. For binary classification a
//! step function might be very useful. For more complex tasks continious activation functions such
//! as Sigmoid, TanH, Softmax or ReLU should be used. In most cases ReLU might prove the best
//! results.
//!
//! The activation function is also sometimes called transfer function.
pub use self::sigmoid::Sigmoid;

pub mod sigmoid;
