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

#[derive(Debug)]
/// Stochastic Gradient Descent with Momentum.
///
/// See [module description][1] for more information.
/// [1]: ./index.html
pub struct Momentum<SolverB: IBackend + SolverOps<f32>> {
    /// The gradient update from the previous iteration for each blob.
    history: Vec<ArcLock<SharedTensor<f32>>>,
    /// The backend used for computing the gradient.
    backend: Rc<SolverB>,

    /// Scalar that temporarily holds learing rate for weight update computations
    lr: SharedTensor<f32>,
    /// Scalar that temporarily holds momentum for weight update computations
    momentum: SharedTensor<f32>,
}

impl<SolverB: IBackend + SolverOps<f32>> Momentum<SolverB> {
    /// Create a new SGD Momentum solver.
    ///
    /// Should not be called directly.
    /// Use [Solver::from_config][2] instead.
    ///
    /// [2]: ../../../solver/struct.Solver.html#method.from_config
    pub fn new(backend: Rc<SolverB>) -> Momentum<SolverB> {
        let (lr, momentum) = {
            let device = IBackend::device(backend.as_ref());

            (SharedTensor::<f32>::new(device, &1).unwrap(),
             SharedTensor::<f32>::new(device, &1).unwrap())
        };
        
        Momentum {
            history: Vec::new(),
            backend: backend,

            lr: lr,
            momentum: momentum,
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
        ::weight::FillerType::Constant {
            value: global_lr * blob_lr
        }.fill(&mut self.lr);

        ::weight::FillerType::Constant {
            value: config.momentum
        }.fill(&mut self.momentum);

        let backend = ISolver::<B, NetB>::backend(self);
        let device = IBackend::device(backend);

        let history_blob = &self.history[history_blob_id];

        let _ = weight_gradient.write().unwrap().add_device(device);
        weight_gradient.write().unwrap().sync(device).unwrap();
        let _ = history_blob.write().unwrap().add_device(device);
        history_blob.write().unwrap().sync(device).unwrap();

        Axpby::axpby_plain(backend,
                           &self.lr,
                           &weight_gradient.read().unwrap(),
                           &self.momentum,
                           &mut history_blob.write().unwrap()).unwrap();

        backend.copy_plain(
            &history_blob.read().unwrap(), &mut weight_gradient.write().unwrap()).unwrap();
    }
}

impl_isolver_sgd!(Momentum<SolverB>);
