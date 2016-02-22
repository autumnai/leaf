//! A [Stochastic Gradient Descent with Momentum][1]
//! [1]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
//!
//! Momentum in solving neural networks works similar to
//! they way it does in physics.
//! If you travel into a a direction with a high velocity,
//! it becomes very hard to change (or reverse)
//! the direction in which you are moving.
//!
//! Similarly when adjusting gradients during solving,
//! keeping a part of the previous gradient update can make solving faster,
//! since if you keep adjusting the gradients
//! into the same direction you will reach the optimum faster.
//! It also makes solving more stable.
use co::backend::*;
use co::tensor::*;
use co::memory::MemoryType;
// use shared_memory::*;
use network::Network;
use solver::*;
use solvers::SGDSolver;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use util::*;

#[derive(Debug, Clone)]
/// Stochastic Gradient Descent with Momentum.
///
/// See [module description][1] for more information.
/// [1]: ./index.html
pub struct Momentum<SolverB: IBackend + SolverOps<f32>> {
    /// The gradient update from the previous iteration for each blob.
    history: Vec<ArcLock<SharedTensor<f32>>>,
    /// The backend used for computing the gradient.
    backend: Rc<SolverB>,
}

impl<SolverB: IBackend + SolverOps<f32>> Momentum<SolverB> {
    /// Create a new SGD Momentum solver.
    ///
    /// Should not be called directly.
    /// Use [Network::from_config][1] or [Solver::from_config][2] instead.
    ///
    /// [1]: ../../../network/struct.Network.html#method.from_config
    /// [2]: ../../../solver/struct.Solver.html#method.from_config
    pub fn new(backend: Rc<SolverB>) -> Momentum<SolverB> {
        Momentum {
            history: Vec::new(),
            backend: backend
        }
    }

    /// Initialize the SGD Momentum solver, allocating memory for its history.
    fn init<B: IBackend + LayerOps<f32>>(&mut self, net: &Network<B>) {
        self.history = Vec::with_capacity(net.learnable_weight_gradients().len());

        for weight_gradient in net.learnable_weight_gradients() {
            let shape = weight_gradient.read().unwrap().desc().clone();
            let history_tensor = Arc::new(RwLock::new(SharedTensor::new(self.backend.device(), &shape).unwrap()));
            self.history.push(history_tensor);
        }
    }
}

impl<B: IBackend + SolverOps<f32>, NetB: IBackend + LayerOps<f32>> SGDSolver<B, NetB> for Momentum<B> {
    fn compute_update_value(&mut self,
                            config: &SolverConfig,
                            weight_gradient: &ArcLock<SharedTensor<f32>>,
                            history_blob_id: usize,
                            global_lr: &f32,
                            blob_lr: &f32) {
        let history_blob = &self.history[history_blob_id];
        let local_momentum = config.momentum;
        let local_lr = global_lr * blob_lr;

        let mut lr_shared = SharedTensor::<f32>::new(self.backend.device(), &1).unwrap();
        if let &mut MemoryType::Native(ref mut lr) = lr_shared.get_mut(self.backend.device()).unwrap() {
            let lr_slice = lr.as_mut_slice::<f32>();
            lr_slice[0] = local_lr;
        } else {
            panic!();
        }

        let mut momentum_shared = SharedTensor::<f32>::new(self.backend.device(), &1).unwrap();
        if let &mut MemoryType::Native(ref mut momentum) = momentum_shared.get_mut(self.backend.device()).unwrap() {
            let momentum_slice = momentum.as_mut_slice::<f32>();
            momentum_slice[0] = local_momentum;
        } else {
            panic!();
        }

        // Compute the update to history, then copy it to the parameter diff.
        let _ = Axpby::<f32>::axpby_plain(ISolver::<B, NetB>::backend(self),
                                               &lr_shared,
                                               &weight_gradient.read().unwrap(),
                                               &momentum_shared,
                                               &mut history_blob.write().unwrap());

        let _ = ISolver::<B, NetB>::backend(self).copy_plain(
            &history_blob.read().unwrap(), &mut weight_gradient.write().unwrap());
    }
}

impl_isolver_sgd!(Momentum<SolverB>);
