# Create a Network

In the previous chapters, we learned that in Leaf everything is build by
layers and that the constructed thing is again a layer, which means it can
function as a new building block for something bigger. This is possible, because
a `Layer` can implement any behavior as long as it takes an input and produces
an output.

In [2.1 Layer Lifecycle](./layer-lifecycle.html)
we have seen, that only one `LayerConfig` can be used to turn it via
`Layer::from_config` into an actual `Layer`. But as Deep Learning relies on
chaining multiple layers together, we need a `Layer`, who implements this
behavior for us.

Enter the container layers.

### Networks via the `Sequential` layer

A `Sequential` Layer is a layer of type container layer. The config of a
container layer has a special method called,
`.add_layer` which takes one `LayerConfig` and adds it to an ordered list in the
`SequentialConfig`.

When turning a `SequentialConfig` into a `Layer` by passing the config to
`Layer::from_config`, the behavior of the `Sequential` is to initialize all the
layers which were added via `.add_layer` and connect the layers with each other.
This means, the output of one layer becomes the input of the next layer in the
list.

The input of a sequential `Layer` becomes the input of the
first layer in the sequential worker, the sequential worker then takes care
of passing the input through all the layers and the output of the last layer
then becomes the output of the `Layer` with the sequential worker. Therefore
a sequential `Layer` fulfills the requirements of a `Layer` - take an input,
return an output.

```rust
// short form for: &LayerConfig::new("net", LayerType::Sequential(cfg))
let mut net_cfg = SequentialConfig::default();

net_cfg.add_input("data", &vec![batch_size, 28, 28]);
net_cfg.add_layer(LayerConfig::new("reshape", ReshapeConfig::of_shape(&vec![batch_size, 1, 28, 28])));
net_cfg.add_layer(LayerConfig::new("conv", ConvolutionConfig { num_output: 20, filter_shape: vec![5], stride: vec![1], padding: vec![0] }));
net_cfg.add_layer(LayerConfig::new("pooling", PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] }));
net_cfg.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 500 }));
net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
net_cfg.add_layer(LayerConfig::new("linear2", LinearConfig { output_size: 10 }));
net_cfg.add_layer(LayerConfig::new("log_softmax", LayerType::LogSoftmax));

// set up the sequential layer aka. a deep, convolutional network
let mut net = Layer::from_config(backend.clone(), &net_cfg);
```

As a sequential layer is like any other layer, we can use sequential layers as
building blocks for larger networks. Important building blocks of a network can
be grouped into a sequential layer and published as a crate for others to use.

```rust
// short form for: &LayerConfig::new("net", LayerType::Sequential(cfg))
let mut conv_net = SequentialConfig::default();

conv_net.add_input("data", &vec![batch_size, 28, 28]);
conv_net.add_layer(LayerConfig::new("reshape", ReshapeConfig::of_shape(&vec![batch_size, 1, 28, 28])));
conv_net.add_layer(LayerConfig::new("conv", ConvolutionConfig { num_output: 20, filter_shape: vec![5], stride: vec![1], padding: vec![0] }));
conv_net.add_layer(LayerConfig::new("pooling", PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] }));
conv_net.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 500 }));
conv_net.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
conv_net.add_layer(LayerConfig::new("linear2", LinearConfig { output_size: 10 }));

let mut net_cfg = SequentialConfig::default();

net_cfg.add_layer(conv_net);
net_cfg.add_layer(LayerConfig::new("linear", LinearConfig { output_size: 500 }));
net_cfg.add_layer(LayerConfig::new("log_softmax", LayerType::LogSoftmax));

// set up the 'big' network
let mut net = Layer::from_config(backend.clone(), &net_cfg);
```

### Networks via other container layers

So far, there is only the sequential layer, but other container layers, with
slightly different behaviors are conceivable. For example a parallel or
concat layer in addition to the sequential layer.

How to 'train' or optimize the constructed network is topic of chapter [3.
Solvers](./solvers.html)
