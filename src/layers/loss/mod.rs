//! Provides methods to calculate the loss (cost) of some output.
//!
//! A loss function is also sometimes called cost function.
#[macro_export]
macro_rules! impl_ilayer_loss {
    () => (
        fn exact_num_output_blobs(&self) -> Option<usize> { Some(1) }
        fn exact_num_input_blobs(&self) -> Option<usize> { Some(1) }
        fn auto_output_blobs(&self) -> bool { true }

        fn loss_weight(&self, output_id: usize) -> Option<f32> {
            if output_id == 0 {
                Some(1f32)
            } else {
                None
            }
        }
    )
}

pub use self::negative_log_likelihood::{NegativeLogLikelihood, NegativeLogLikelihoodConfig};

pub mod negative_log_likelihood;
