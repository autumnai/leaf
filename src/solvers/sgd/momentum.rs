//! TODO
use math::*;
use shared_memory::*;
use network::Network;
use solver::*;
use solvers::SGDSolver;

#[derive(Debug, Clone)]
/// Stochastic Gradient Descent with Momentum
pub struct Momentum {
    /// The gradient update from the previous iteration for each blob.
    history: Vec<ArcLock<HeapBlob>>,
}

impl Momentum {
    /// TODO
    pub fn new() -> Momentum {
        Momentum {
            history: Vec::new(),
        }
    }

    fn init(&mut self, net: &Network) {
        self.history = Vec::with_capacity(net.learnable_weights().len());

        for weight_blob in net.learnable_weights() {
            let shape = weight_blob.read().unwrap().shape();
            let history_blob = new_shared_heapblob();
            history_blob.write().unwrap().reshape(shape);
            self.history.push(history_blob);
        }
    }
}

impl SGDSolver for Momentum {
    fn compute_update_value(&mut self,
                            config: &SolverConfig,
                            weight_blob: &ArcLock<HeapBlob>,
                            history_blob_id: usize,
                            global_lr: &f32,
                            blob_lr: &f32) {
        let history_blob = &self.history[history_blob_id];
        let momentum = config.momentum;
        let local_lr = global_lr * blob_lr;

        // Compute the update to history, then copy it to the parameter diff.
        leaf_cpu_axpby(&local_lr,
                       weight_blob.read().unwrap().cpu_diff(),
                       &momentum,
                       history_blob.write().unwrap().mutable_cpu_data());
        *weight_blob.write().unwrap().mutable_cpu_diff() =
            history_blob.read().unwrap().cpu_data().clone();
    }
}

impl_isolver_sgd!(Momentum);
