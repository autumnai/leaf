//! Computes the multinomial logistic loss of the softmax of its bottom Blob.
//!
//! This is conceptually identical to a softmax layer followed by a multinomial
//! logistic loss layer, but provides a more numerically stable gradient.

#[derive(Debug, Copy, Clone)]
/// Softmax Loss Layer
pub struct Softmax;
