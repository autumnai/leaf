//! Provides the generics and interfaces for the specific Solvers.
//!
//! See [Solvers][solvers]
//! [solvers]: ../solvers/index.html

pub mod confusion_matrix;

pub use self::confusion_matrix::ConfusionMatrix;

use std::rc::Rc;
use std::marker::PhantomData;
use co::prelude::*;
use layer::*;
use layers::SequentialConfig;
use solvers::*;
use util::{ArcLock, LayerOps, SolverOps};

#[derive(Debug)]
/// Solver that optimizes a [Layer][1] with a given objective.
/// [1]: ../layer/index.html
pub struct Solver<SolverB: IBackend + SolverOps<f32>, B: IBackend + LayerOps<f32>> {
    net: Layer<B>,
    objective: Layer<SolverB>,
    /// The implementation of the Solver
    pub worker: Box<ISolver<SolverB, B>>,

    config: SolverConfig,

    /// The current iteration / number of times weights have been updated
    iter: usize,

    solver_backend: PhantomData<SolverB>,
}

impl<SolverB: IBackend + SolverOps<f32> + 'static, B: IBackend + LayerOps<f32> + 'static> Solver<SolverB, B> {
    /// Create Solver from [SolverConfig][1]
    /// [1]: ./struct.SolverConfig.html
    ///
    /// This is the **preferred method** to create a Solver for training a neural network.
    pub fn from_config(net_backend: Rc<B>, obj_backend: Rc<SolverB>, config: &SolverConfig) -> Solver<SolverB, B> {
        let network = Layer::from_config(net_backend, &config.network);
        let mut worker = config.solver.with_config(obj_backend.clone(), &config);
        worker.init(&network);

        Solver {
            worker: worker,
            net: network,
            objective: Layer::from_config(obj_backend, &config.objective),
            iter: 0,

            config: config.clone(),
            solver_backend: PhantomData::<SolverB>,
        }
    }

}

impl<SolverB: IBackend + SolverOps<f32> + 'static, B: IBackend + LayerOps<f32> + 'static> Solver<SolverB, B>{
    fn init(&mut self, backend: Rc<B>) {
        info!("Initializing solver from configuration");

        let mut config = self.config.clone();
        self.init_net(backend, &mut config);
    }

    /// Initialize the training net
    fn init_net(&mut self, backend: Rc<B>, param: &mut SolverConfig) {
        self.net = Layer::from_config(backend, &param.network);
    }

    /// Train the network with one minibatch
    pub fn train_minibatch(&mut self, mb_data: ArcLock<SharedTensor<f32>>, mb_target: ArcLock<SharedTensor<f32>>) -> ArcLock<SharedTensor<f32>> {
        self.net.clear_weights_gradients();

        // forward through network and classifier
        let network_out = self.net.forward(&[mb_data])[0].clone();
        let _ = self.objective.forward(&[network_out.clone(), mb_target]);

        // forward through network and classifier
        let classifier_gradient = self.objective.backward(&[]);
        self.net.backward(&classifier_gradient[0 .. 1]);

        self.worker.compute_update(&self.config, &mut self.net, self.iter);
        self.net.update_weights(self.worker.backend());
        self.iter += 1;

        network_out
    }

    /// Returns the network trained by the solver.
    ///
    /// This is the recommended method to get a usable trained network.
    pub fn network(&self) -> &Layer<B> {
        &self.net
    }

    /// Returns the network trained by the solver.
    ///
    /// This is the recommended method to get a trained network,
    /// if you want to alter the network. Keep in mind that altering the network
    /// might render the solver unusable and continuing training the network with it will yield
    /// unexpected results.
    pub fn mut_network(&mut self) -> &mut Layer<B> {
        &mut self.net
    }
}

/// Implementation of a specific Solver.
///
/// See [Solvers][1]
/// [1]: ../solvers/index.html
pub trait ISolver<SolverB, B: IBackend + LayerOps<f32>> {
    /// Initialize the solver, setting up any network related data.
    fn init(&mut self, net: &Layer<B>) {}

    /// Update the weights of the net with part of the gradient.
    ///
    /// The [second phase of backpropagation learning][1].
    /// Calculates the gradient update that should be applied to the network,
    /// and then applies that gradient to the network, changing its weights.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation#Phase_2:_Weight_update
    ///
    /// Used by [step][2] to optimize the network.
    ///
    /// [2]: ./struct.Solver.html#method.step
    fn compute_update(&mut self, param: &SolverConfig, network: &mut Layer<B>, iter: usize);

    /// Returns the backend used by the solver.
    fn backend(&self) -> &SolverB;
}

impl<SolverB, B: IBackend + LayerOps<f32>> ::std::fmt::Debug for ISolver<SolverB, B> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "({})", "ILayer")
    }
}

#[derive(Debug, Clone)]
/// Configuration for a Solver
pub struct SolverConfig {
    /// Name of the solver.
    pub name: String,
    /// The [LayerConfig][1] that is used to initialize the network.
    /// [1]: ../layer/struct.LayerConfig.html
    pub network: LayerConfig,
    /// The [LayerConfig][1] that is used to initialize the objective.
    /// [1]: ../layer/struct.LayerConfig.html
    pub objective: LayerConfig,
    /// The [Solver implementation][1] to be used.
    /// [1]: ../solvers/index.html
    pub solver: SolverKind,
    /// Accumulate gradients over `minibatch_size` instances.
    ///
    /// Default: 1
    pub minibatch_size: usize,
    /// The learning rate policy to be used.
    ///
    /// Default: Fixed
    pub lr_policy: LRPolicy,
    /// The base learning rate.
    ///
    /// Default: 0.01
    pub base_lr: f32,
    /// gamma as used in the calculation of most learning rate policies.
    ///
    /// Default: 0.1
    pub gamma: f32,
    /// The stepsize used in Step and Sigmoid learning policies.
    ///
    /// Default: 10
    pub stepsize: usize,
    /// The threshold for clipping gradients.
    ///
    /// Gradient values will be scaled to their [L2 norm][1] of length `clip_gradients`
    /// if their L2 norm is larger than `clip_gradients`.
    /// If set to `None` gradients will not be clipped.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
    ///
    /// Default: None
    pub clip_gradients: Option<f32>,
    /// The global [weight decay][1] multiplier for [regularization][2].
    /// [1]: http://www.alglib.net/dataanalysis/improvinggeneralization.php#header3
    /// [2]: https://cs231n.github.io/neural-networks-2/#reg
    ///
    /// Regularization can prevent [overfitting][3].
    ///
    /// If set to `None` no regularization will be performed.
    ///
    /// [3]: https://cs231n.github.io/neural-networks-2/#reg
    pub weight_decay: Option<f32>,
    /// The method of [regularization][1] to use.
    /// [1]: https://cs231n.github.io/neural-networks-2/#reg
    ///
    /// There are different methods for regularization.
    /// The two most common ones are [L1 regularization][1] and [L2 regularization][1].
    ///
    /// See [RegularizationMethod][2] for all implemented methods.
    ///
    /// [2]: ./enum.RegularizationMethod.html
    ///
    /// Currently only L2 regularization is implemented.
    /// See [Issue #23](https://github.com/autumnai/leaf/issues/23).
    pub regularization_method: Option<RegularizationMethod>,
    /// The [momentum][1] multiplier for [SGD solvers][2].
    /// [1]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
    /// [2]: ../solvers/sgd/index.html
    ///
    /// For more information see [SGD with momentum][3]
    /// [3]: ../solvers/sgd/momentum/index.html
    ///
    /// The value should always be between 0 and 1 and dictates how much of the previous
    /// gradient update will be added to the current one.
    ///
    /// Default: 0
    pub momentum: f32,
}

impl Default for SolverConfig {
    fn default() -> SolverConfig {
        SolverConfig {
            name: "".to_owned(),
            network: LayerConfig::new("default", SequentialConfig::default()),
            objective: LayerConfig::new("default", SequentialConfig::default()),
            solver: SolverKind::SGD(SGDKind::Momentum),

            minibatch_size: 1,

            lr_policy: LRPolicy::Fixed,
            base_lr: 0.01f32,
            gamma: 0.1f32,
            stepsize: 10,

            clip_gradients: None,

            weight_decay: None,
            regularization_method: None,

            momentum: 0f32,
        }
    }
}

impl SolverConfig {
    /// Return the learning rate for a supplied iteration.
    ///
    /// The way the learning rate is calculated depends on the configured [LRPolicy][1].
    ///
    /// [1]: ./enum.LRPolicy.html
    ///
    /// Used by the [Solver][2] to calculate the learning rate for the current iteration.
    /// The calculated learning rate has a different effect on training dependent on what
    /// [type of Solver][3] you are using.
    ///
    /// [2]: ./struct.Solver.html
    /// [3]: ../solvers/index.html
    pub fn get_learning_rate(&self, iter: usize) -> f32 {
        match self.lr_policy() {
            LRPolicy::Fixed => {
                self.base_lr()
            }
            LRPolicy::Step => {
                let current_step = self.step(iter);
                self.base_lr() * self.gamma().powf(current_step as f32)
            }
            // LRPolicy::Multistep => {
            //     // TODO: the current step can be calculated on-demand
            //     //   if (this->current_step_ < this->param_.stepvalue_size() &&
            //     //         this->iter_ >= this->param_.stepvalue(this->current_step_)) {
            //     //     this->current_step_++;
            //     //     LOG(INFO) << "MultiStep Status: Iteration " <<
            //     //     this->iter_ << ", step = " << this->current_step_;
            //     //   }
            //     //   rate = this->param_.base_lr() *
            //     //       pow(this->param_.gamma(), this->current_step_);
            //     unimplemented!();
            // }
            LRPolicy::Exp => {
                self.base_lr() * self.gamma().powf(iter as f32)
            }
            // LRPolicy::Inv => {
            //     //   rate = this->param_.base_lr() *
            //     //       pow(Dtype(1) + this->param_.gamma() * this->iter_,
            //     //           - this->param_.power());
            //     unimplemented!();
            // }
            // LRPolicy::Poly => {
            //     //   rate = this->param_.base_lr() * pow(Dtype(1.) -
            //     //       (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
            //     //       this->param_.power());
            //     unimplemented!();
            // }
            // LRPolicy::Sigmoid => {
            //     //   rate = this->param_.base_lr() * (Dtype(1.) /
            //     //       (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
            //     //         Dtype(this->param_.stepsize())))));
            //     unimplemented!();
            // }
        }
    }

    /// Return current step at iteration `iter`.
    ///
    /// Small helper for learning rate calculation.
    fn step(&self, iter: usize) -> usize {
        iter / self.stepsize()
    }

    /// Return learning rate policy.
    fn lr_policy(&self) -> LRPolicy {
        self.lr_policy
    }

    /// Return the base learning rate.
    fn base_lr(&self) -> f32 {
        self.base_lr
    }

    /// Return the gamma for learning rate calculations.
    fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Return the stepsize for learning rate calculations.
    fn stepsize(&self) -> usize {
        self.stepsize
    }
}

#[derive(Debug, Copy, Clone)]
/// All available types of solvers.
pub enum SolverKind {
    /// Stochastic Gradient Descent.
    /// See [SGDKind][1] for all available SGD solvers.
    /// [1]: ./enum.SGDKind.html
    SGD(SGDKind),
}

impl SolverKind {
    /// Create a Solver of the specified kind with the supplied SolverConfig.
    pub fn with_config<B: IBackend + SolverOps<f32> + 'static, NetB: IBackend + LayerOps<f32> + 'static>(&self, backend: Rc<B>, config: &SolverConfig) -> Box<ISolver<B, NetB>> {
        match *self {
            SolverKind::SGD(sgd) => {
                sgd.with_config(backend, config)
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// All available types of Stochastic Gradient Descent solvers.
pub enum SGDKind {
    /// Stochastic Gradient Descent with Momentum. See [implementation][1]
    /// [1] ../solvers/
    Momentum,
}

impl SGDKind {
    /// Create a Solver of the specified kind with the supplied SolverConfig.
    pub fn with_config<B: IBackend + SolverOps<f32> + 'static, NetB: IBackend + LayerOps<f32> + 'static>(&self, backend: Rc<B>, config: &SolverConfig) -> Box<ISolver<B, NetB>> {
        match *self {
            SGDKind::Momentum => {
                Box::new(Momentum::<B>::new(backend))
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Learning Rate Policy for a [Solver][1]
/// [1]: ./struct.Solver.html
///
/// The variables mentioned below are defined in the [SolverConfig][2] apart from
/// iter, which is the current iteration of the solver, that is supplied as a parameter
/// for the learning rate calculation.
///
/// [2]: ./struct.SolverConfig.html
pub enum LRPolicy {
    /// always return base_lr
    Fixed,
    /// learning rate decays every `step` iterations.
    /// return base_lr * gamma ^ (floor(iter / step))
    Step,
    // /// similar to step but it allows non uniform steps defined by
    // /// stepvalue
    // Multistep,
    /// return base_lr * gamma ^ iter
    Exp,
    // /// return base_lr * (1 + gamma * iter) ^ (- power)
    // Inv,
    // /// the effective learning rate follows a polynomial decay, to be
    // /// zero by the max_iter.
    // /// return base_lr (1 - iter/max_iter) ^ (power)
    // Poly,
    // /// the effective learning rate follows a sigmod decay
    // /// return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
    // Sigmoid,
}

#[derive(Debug, Copy, Clone)]
/// [Regularization][1] method for a [Solver][2].
/// [1]: https://cs231n.github.io/neural-networks-2/#reg
/// [2]: ./struct.Solver.html
pub enum RegularizationMethod {
    /// L2 regularization
    L2,
}
