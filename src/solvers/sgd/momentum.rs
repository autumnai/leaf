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
use co::prelude::*;
use coblas::plugin::Copy;
use layer::*;
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

}

impl<B: IBackend + SolverOps<f32>, NetB: IBackend + LayerOps<f32> + 'static> SGDSolver<B, NetB> for Momentum<B> {
    fn compute_update_value(&mut self,
                            config: &SolverConfig,
                            weight_gradient: &ArcLock<SharedTensor<f32>>,
                            history_blob_id: usize,
                            global_lr: &f32,
                            blob_lr: &f32) {
        let history_blob = &self.history[history_blob_id];
        let local_momentum = config.momentum;
        let local_lr = global_lr * blob_lr;

        let native_backend = native_backend();
        let backend = ISolver::<B, NetB>::backend(self);
        let device = IBackend::device(backend);

        let lr_shared = native_scalar(local_lr);
        let momentum_shared = native_scalar(local_momentum);

        let _ = weight_gradient.write().unwrap().add_device(native_backend.device());
        weight_gradient.write().unwrap().sync(native_backend.device()).unwrap();
        let _ = history_blob.write().unwrap().add_device(native_backend.device());
        history_blob.write().unwrap().sync(native_backend.device()).unwrap();
        Axpby::<f32>::axpby_plain(&native_backend,
                                               &lr_shared,
                                               &weight_gradient.read().unwrap(),
                                               &momentum_shared,
                                               &mut history_blob.write().unwrap()).unwrap();

        native_backend.copy_plain(
            &history_blob.read().unwrap(), &mut weight_gradient.write().unwrap()).unwrap();
    }
}

impl_isolver_sgd!(Momentum<SolverB>);
