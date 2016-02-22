//! Flattens the bottom Blob into a simpler top Blob.
//!
//! Input of shape n * c * h * w becomes
//! a simple vector output of shape n * (c*h*w).
//!
#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Flattening Utility Layer
pub struct Flatten;
