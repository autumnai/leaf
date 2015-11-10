//! Provides the trainers for the [Network][network].
//! [network]: ../network/index.html

#[allow(unused_import_braces)]
pub use self::sgd::{Momentum};
pub mod sgd;


use math::*;
use shared_memory::*;
use solver::*;
use network::Network;

trait SGDSolver {
    fn compute_update_value(&mut self,
                            config: &SolverConfig,
                            weight_blob: &ArcLock<HeapBlob>,
                            history_blob_id: usize,
                            global_lr: &f32,
                            blob_lr: &f32);

    /// [Clip gradients][1] when they exceed [SolverConfig.clip_gradients][2].
    /// [1]: http://arxiv.org/abs/1211.5063
    /// [2]: ../solver/struct.SolverConfig.html
    ///
    /// [Gradient norm clipping][1] is a technique used when dealing with
    /// [Recurrent Neural Networks][3].
    /// When the [L2 norm][4] of the gradients exceeds a threshold it is "clipped"
    /// to that threshold. The naming can be misleading since the gradients are not
    /// actually clipped (as in cut off), but rescaled to the threshold.
    ///
    /// [3]: https://en.wikipedia.org/wiki/Recurrent_neural_network
    /// [4]: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
    fn clip_gradients(&self, config: &SolverConfig, net: &mut Network) {
        // skip clipping gradients if SolverConfig.clip_gradients is set to None
        if let Some(clip_threshold) = config.clip_gradients {
            let net_weights = net.learnable_weights();
            let mut sumsq_diff = 0f32;
            for weight_blob in net_weights {
                let blob = weight_blob.read().unwrap();
                let blob_sumsq_diff = leaf_cpu_dot(blob.cpu_diff(), blob.cpu_diff());
                sumsq_diff += blob_sumsq_diff;
            }
            let l2norm_diff = sumsq_diff.sqrt();
            if l2norm_diff > clip_threshold {
                let scale_factor = clip_threshold / l2norm_diff;
                info!("Gradient clipping: scaling down gradients (L2 norm {} > {})
                        by scale factor {}",
                    l2norm_diff,
                    clip_threshold,
                    scale_factor);

                for weight_blob in net_weights {
                    let mut blob = weight_blob.write().unwrap();
                    let diff = blob.mutable_cpu_diff();
                    leaf_cpu_scal(&scale_factor, diff);
                }
            }
        }
    }

    /// Scale the gradient to counteract the [SolverConfig.minibatch_size][1]
    /// [1]: ../solver/struct.SolverConfig.html
    ///
    /// To counteract that we are accumulating the gradients over multiple samples,
    /// we need to scale the gradients down to the equivalent of a single sample.</br>
    /// E.g. with a `minibatch_size` of 4 we need to scale the gradient by 0.25 (= 1/4).
    fn normalize(&self, config: &SolverConfig, weight_blob: &ArcLock<HeapBlob>) {
        if config.minibatch_size > 1 {
            let scale_factor = 1f32 / config.minibatch_size as f32;
            let mut write_blob = weight_blob.write().unwrap();
            leaf_cpu_scal(&scale_factor, write_blob.mutable_cpu_diff());
        }
    }

    /// [Regularize][1] the gradient according to the configured [RegularizationMethod][2].
    /// [1]: https://cs231n.github.io/neural-networks-2/#reg
    /// [2]: ../solver/enum.RegularizationMethod.html
    fn regularize(&self, config: &SolverConfig, weight_blob: &ArcLock<HeapBlob>, blob_weight_decay: Option<f32>) {
        if let Some(global_weight_decay) = config.weight_decay {
            if let Some(regularization_method) = config.regularization_method {
                match blob_weight_decay {
                    Some(weight_decay_mult) => {
                        let local_decay = global_weight_decay * weight_decay_mult;
                        match regularization_method {
                            RegularizationMethod::L2 => {
                                leaf_cpu_axpy(&local_decay,
                                              weight_blob.read().unwrap().cpu_data(),
                                              weight_blob.write().unwrap().mutable_cpu_diff());
                            },
                        }
                    },
                    None => {
                        error!("Weight decay multiplier for blob missing.");
                    },
                }
            }
        }
    }
}
