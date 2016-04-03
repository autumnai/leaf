# Layer Lifecycle

In [2. Layers](./layers.html) we have already seen a little bit about how to
construct a `Layer` from a `LayerConfig`. In this chapter, we take
a closer look at what happens inside Leaf when initializing a `Layer` when
running the `.forward` of a `Layer` and when running the `.backward`. In the
next chapter [2.2 Create a Network](./building-networks.html) we then
apply our knowledge to construct deep networks via the container layer.

Initialization (`::from_config`), `.forward` and `.backward` are the three most
important methods of a `Layer` and describe basically the entire API. Let's
take a closer look at what happens inside Leaf, when these methods are called.

### Initialization

A layer is constructed from a `LayerConfig` via the `Layer::from_config`
method, which returns a fully initialized `Layer`.

```rust
let mut sigmoid: Layer = Layer::from_config(backend.clone(), &LayerConfig::new("sigmoid", LayerType::Sigmoid))
let mut alexnet: Layer = Layer::from_config(backend.clone(), &LayerConfig::new("alexnet", LayerType::Sequential(cfg)))
```

In the example above, the first layer has a Sigmoid worker
(`LayerType::Sigmoid`). The second layer has a Sequential worker.
Although both `Layer::from_config` methods, return a `Layer`, the behavior of
the `Layer` depends on the `LayerConfig` it was constructed with. The
`Layer::from_config` calls internally the `worker_from_config` method, which
constructs the specific worker defined by the `LayerConfig`.

```rust
fn worker_from_config(backend: Rc<B>, config: &LayerConfig) -> Box<ILayer<B>> {
    match config.layer_type.clone() {
        // more matches
        LayerType::Pooling(layer_config) => Box::new(Pooling::from_config(&layer_config)),
        LayerType::Sequential(layer_config) => Box::new(Sequential::from_config(backend, &layer_config)),
        LayerType::Softmax => Box::new(Softmax::default()),
        // more matches
    }
}
```

The layer specific `::from_config` (if available or needed) then takes care of
initializing the worker struct, allocating memory for weights and so on.

In case the worker layer is a container layer, its `::from_config` takes
care of initializing all the `LayerConfig`s it contains (which were added via its
`.add_layer` method) and connecting them in
the order they were provided to the `LayerConfig` of the container.

Every `.forward` or `.backward` call that is now made to the returned `Layer` is
sent to the worker.

### Forward

The `forward` method of a `Layer` sends the input through the constructed
network and returns the output of the network's final layer.

The `.forward` method does three things:

1. Reshape the input data if necessary
2. Sync the input/weights to the device were the computation happens. This step
removes the worker layer from the obligation to care about memory synchronization.
3. Call the `forward` method of the worker layer.

In case, the worker layer is a container layer, the `.forward` method of the
container layer takes care of calling the `.forward` methods of its managed
layers in the right order.

### Backward

The `.backward` of a `Layer` works quite similar to its `.forward`. Although it
does not need to reshape the input. The `.backward` computes
the gradient with respect to the input and the gradient w.r.t. the parameters but
only returns the gradient w.r.t the input as only that is needed to compute the
gradient of the entire network via the chain rule.

In case the worker layer is a container layer, the `.backward` method of the
container layer takes care of calling the `.backward_input` and
`.backward_parameter` methods of its managed layers in the right order.
