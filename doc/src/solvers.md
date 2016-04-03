# Solvers

Solvers optimize the layer with a given objective. This might happen
by updating the weights of the layer, which is the usual practice for
Neural Networks but is not limited to this kind of learning.

A solver can have different learning (solving) policies. With Neural Networks, it
is common to use a Stochastic Gradient Descent based approach
like Adagrad, whereas for a classical regression the solving might be
done via a maximum likelihood estimation.

Similar to `Layer`s, we can construct a [`Solver` (_/src/solver/mod.rs_)][solver]
from a [`SolverConfig` (_/src/solver/mod.rs_)][solver-config].
When passing this `SolverConfig` (e.g. an Adagrad `SolverConfig`) to the
`Solver::from_config` method, a `Solver` with the behavior
of the config is returned.

The most characteristic feature of the `SolverConfig` is its `network`
and `objective` fields. These two fields expect one `LayerConfig` each. When
passing the `SolverConfig` to the `Solver::from_config` method, the
`LayerConfig` of the `network` and `objective` fields are turned into
an initialized `Layer` and provided to the returned, `Solver`.

```rust
// set up a Solver
let mut solver_cfg = SolverConfig { minibatch_size: batch_size, base_lr: learning_rate, momentum: momentum, .. SolverConfig::default() };
solver_cfg.network = LayerConfig::new("network", net_cfg);
solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);
```

The now initialized `Solver` can be feed with data to optimize the `network`.

[solver]: https://github.com/autumnai/leaf/blob/master/src/solver/mod.rs
[solver-config]: https://github.com/autumnai/leaf/blob/master/src/solver/mod.rs
