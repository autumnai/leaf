//! Provides [ISolver][1] implementations based on [Stochastic Gradient Descent][2].
//! [1]: ../solver/trait.ISolver.html
//! [2]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

/// Implement [ISolver][1] for [SGD solvers][2].
/// [1]: ./solver/trait.ISolver.html
/// [2]: ./solvers/sgd/index.html
#[macro_export]
macro_rules! impl_isolver_sgd {
    ($t:ty) => (
        impl ISolver for $t {
            fn apply_update(&mut self, config: &SolverConfig, net: &mut Network, iter: usize) {
                // CHECK(Caffe::root_solver()); // Caffe
                let rate = config.get_learning_rate(iter);

                self.clip_gradients(config, net);
                for (weight_id, weight_blob) in net.learnable_weights().iter().enumerate() {
                    self.normalize(config, weight_blob);
                    self.regularize(config, weight_blob, net.weights_weight_decay()[weight_id]);

                    self.compute_update_value(config,
                                              weight_blob,
                                              weight_id,
                                              &rate,
                                              &net.weights_lr()[weight_id].unwrap());
                }
                net.update_weights();
            }
        }
    )
}

pub use self::momentum::Momentum;

pub mod momentum;
