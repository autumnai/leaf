//! Provides container layers.
//!
//! For now layers in container should be discribed as layers that are used
//! to connect multiple layers together to create 'networks'.
#[macro_export]
macro_rules! impl_ilayer_common {
    () => (
        fn exact_num_output_blobs(&self) -> Option<usize> { Some(1) }
        fn exact_num_input_blobs(&self) -> Option<usize> { Some(1) }
    )
}

pub use self::sequential::{Sequential, SequentialConfig};

pub mod sequential;
