//! Provides ISolver implementations based on [Stochastic Gradient
//! Descent][2].
//! [2]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
//!
//! One of the steps during [backpropagation][backprop] is determining the
//! gradient for each weight.
//! In theory this can be achieved very well using [Gradient Descent (GD)][gd].
//! In practice however applying GD when minimizing an objective function
//! for large datasets, quickly becomes computationaly unfeasible.
//!
//! Luckily GD can be approximated by taking a small random subset
//! from the training dataset, commonly refered to as *mini-batch*.
//! We then compute the gradients
//! only for the samples from the mini-batch and average over these gradients,
//! resulting in an estimate of the global gradient.</br>
//! This method is refered to as **Stochastic Gradient Descent**.
//!
//! [backprop]: https://en.wikipedia.org/wiki/Backpropagation
//! [gd]: https://en.wikipedia.org/wiki/Gradient_descent

/// Implement [ISolver][1] for [SGD solvers][2].
/// [1]: ./solver/trait.ISolver.html
/// [2]: ./solvers/sgd/index.html
#[macro_export]
macro_rules! impl_isolver_sgd {
    ($t:ty) => (
        impl<SolverB: IBackend + SolverOps<f32>, NetB: IBackend + LayerOps<f32> + 'static> ISolver<SolverB, NetB> for $t {
            /// Initialize the SGD Momentum solver, allocating memory for its history.
            fn init(&mut self, net: &Layer<NetB>) {
                self.history = Vec::with_capacity(net.learnable_weights_gradients().len());

                for weight_gradient in net.learnable_weights_gradients() {
                    let shape = weight_gradient.read().unwrap().desc().clone();
                    let mut tensor = SharedTensor::new(&shape);

                    let filler = ::weight::FillerType::Constant { value: 0f32 };
                    filler.fill(&mut tensor);

                    let history_tensor = Arc::new(RwLock::new(tensor));
                    self.history.push(history_tensor);
                }
            }

            fn compute_update(&mut self, config: &SolverConfig, net: &mut Layer<NetB>, iter: usize) {
                let rate = config.get_learning_rate(iter);

                SGDSolver::<SolverB, NetB>::clip_gradients(self, config, net);
                for (weight_id, weight_gradient) in net.learnable_weights_gradients().iter().enumerate() {
                    SGDSolver::<SolverB, NetB>::normalize(self, config, weight_gradient);
                    // SGDSolver::<SolverB, NetB>::regularize(self, config, weight_gradient, net.weights_weight_decay()[weight_id]);

                    SGDSolver::<SolverB, NetB>::compute_update_value(self, config,
                                              weight_gradient,
                                              weight_id,
                                              &rate,
                                              &net.learnable_weights_lr()[weight_id].unwrap());
                }
            }

            fn backend(&self) -> &SolverB {
                &self.backend
            }
        }
    )
}

pub use self::momentum::Momentum;

pub mod momentum;
