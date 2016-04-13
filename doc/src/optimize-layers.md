# Optimize Layers

In the previous chapter [3. Solver](./solvers.html), we learned what a solver
is and what it does. In this chapter, we take a look on how to optimize a
network via a `Solver`.

A `Solver` after its initialization has two `Layer`s, one for the `network`
which will be optimized and one for the `objective`. The output of the `network`
layer is used by the `objective` to compute the loss. The loss is then used by
the `Solver` to optimize the `network`.

The `Solver` has a very simple API - `.train_minibatch` and `.network`. The
optimization of the `network` is kicked off by the `.train_minibatch`
method, which takes two input parameters - some data that is feed to the network
and the expected target value for the network.

A SGD (Stochastic Gradient Descent) `Solver` would now compute the output of
the `network` using as input the data, put the output together with the expected
target value into the `objective` layer and use it, together with the gradient
of the `network` to optimize the weights of the `network`.

```rust
/// Train the network with one minibatch
pub fn train_minibatch(&mut self, mb_data: ArcLock<SharedTensor<f32>>, mb_target: ArcLock<SharedTensor<f32>>) -> ArcLock<SharedTensor<f32>> {
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
```

Using the `.train_minibatch` is straight forward. We pass the data as well as the
expected result of the `network` to the `.train_minibatch` method of the
initialized `Solver` struct. A more detailed example can be found at the
[autumnai/leaf-examples](https://github.com/autumnai/leaf-examples) repository.

```rust
let inp_lock = Arc::new(RwLock::new(inp));
let label_lock = Arc::new(RwLock::new(label));

// train the network!
let inferred_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());
```

If we don't want the `network` to be trained, we can use the `.network` method
of the `Solver` to receive access to the network. The `Solver` has actually
two network methods - `.network` and `mut_network`.

To run just the forward of the `network` without any optimization we can run

```rust
let inferred_out = solver.network().forward(inp_lock.clone());
```

Leaf ships with a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix),
which is a convenient way to visualize the performance of the optimized
`network`.

```rust
let inferred_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

let mut inferred = inferred_out.write().unwrap();
let predictions = confusion.get_predictions(&mut inferred);

confusion.add_samples(&predictions, &targets);
println!("Last sample: {} | Accuracy {}", confusion.samples().iter().last().unwrap(), confusion.accuracy());
```

A more detailed example can be found at the
[autumnai/leaf-examples](https://github.com/autumnai/leaf-examples) repository.
