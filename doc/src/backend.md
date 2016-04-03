# Backend

Via the concept of a backend we can abstract over the platform we will execute
or optimize a network on. The construction of a backend is trivial. The backend
is passed to the `Solver`, (one backend for `network` and one for the
`objectve`). The Solver than executes all operations on the provided backend.

```rust
let backend = ::std::rc::Rc::new(Backend::<Cuda>::default().unwrap());

// set up solver
let mut solver_cfg = SolverConfig { minibatch_size: batch_size, base_lr: learning_rate, momentum: momentum, .. SolverConfig::default() };
solver_cfg.network = LayerConfig::new("network", net_cfg);
solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);
```

The backend is a concept of
[Collenchyma](https://github.com/autumnai/collenchyma), to which you can refer
for now, until this chapter becomes more fleshed out.
