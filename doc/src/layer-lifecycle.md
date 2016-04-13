# Layer Lifecycle

In chapter [2. Layers](./layers.html) we saw how to
construct a simple `Layer` from a `LayerConfig`. In this chapter, we take
a closer look at what happens inside Leaf when initializing a `Layer` and when running its 
`.forward` and `.backward` methods. In the next chapter [2.2 Create a Network](./building-networks.html) we 
apply our knowledge to construct deep networks with the container layer.

The most important methods of a `Layer` are initialization (`::from_config`), `.forward` and `.backward`.
They basically describe the entire API, so let's take a closer look at what happens inside Leaf when these methods are called.

### Initialization

A layer is constructed from a `LayerConfig` with the `Layer::from_config`
method, which returns a fully initialized `Layer`.

```rust
let mut sigmoid: Layer = Layer::from_config(backend.clone(), &LayerConfig::new("sigmoid", LayerType::Sigmoid))
let mut alexnet: Layer = Layer::from_config(backend.clone(), &LayerConfig::new("alexnet", LayerType::Sequential(cfg)))
```

In the example above, the first layer has a Sigmoid worker
(`LayerType::Sigmoid`) and the second layer has a Sequential worker.
Although both `::from_config` methods return a `Layer`, the behavior of
that `Layer` depends on the `LayerConfig` it was constructed with. The
`Layer::from_config` internally calls the `worker_from_config` method, which
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

The layer-specific `::from_config` (if available or needed) then takes care of
initializing the worker struct, allocating memory for weights and so on.

If the worker is a container layer, its `::from_config` takes
care of initializing all the `LayerConfig`s it contains (which were added via its
`.add_layer` method) and connecting them in the order they were provided.

Every `.forward` or `.backward` call that is made on the returned `Layer` is
run by the internal worker.

### Forward

The `forward` method of a `Layer` threads the input through the constructed
network and returns the output of the network's final layer.

The `.forward` method does three things:

1. Reshape the input data if necessary
2. Sync the input/weights to the device where the computation happens. This step
removes the need for the worker layer to care about memory synchronization.
3. Call the `forward` method of the internal worker layer.

If the worker layer is a container layer, the `.forward` method 
takes care of calling the `.forward` methods of its managed
layers in the right order.

### Backward

The `.backward` method of a `Layer` works similarly to `.forward`, apart from
needing to reshape the input. The `.backward` method computes
the gradient with respect to the input as well as the gradient w.r.t. the parameters. However, 
the method only returns the input gradient because that is all that is needed to compute the
gradient of the entire network via the chain rule.

If the worker layer is a container layer, the `.backward` method 
takes care of calling the `.backward_input` and
`.backward_parameter` methods of its managed layers in the right order.
