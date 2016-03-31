# Solvers

Solvers optimize the layer with a given objective. This might happen
by updating the weights of the layer, which is the usual practice for
Neural Networks but is not limited to this kind of learning.

A solver can have different learning (solving) policies. With Neural Networks it
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
`LayerConfig`s for `network` and `objective` are turned into `Layer`s and
provided as fields to the returned, `Solver`.

The now initialized `Solver` can be feed with data to optimize the `network`.

[solver]: https://github.com/autumnai/leaf/blob/master/src/solver/mod.rs
[solver-config]: https://github.com/autumnai/leaf/blob/master/src/solver/mod.rs
