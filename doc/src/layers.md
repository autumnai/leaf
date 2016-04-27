# Layers

### What is a Layer?

[Layers](./deep-learning-glossary.html#Layer) are the only building
blocks in Leaf. As we will see later on, everything is a layer. Even when 
we construct [networks](./deep-learning-glossary.html#Network), we are still just 
working with layers composed of smaller layers. This makes the API clean and expressive.

A layer is like a function: given an input it computes an output. 
It could be some mathematical expression, like Sigmoid, ReLU, or a non-mathematical instruction, 
like querying data from a database, logging data, or anything in between.
In Leaf, layers describe not only the interior 'hidden layers' but also the input and
output layer. 

Layers in Leaf are only slightly opinionated, they need to take
an input and produce an output. This is required in order to successfully stack
layers on top of each other to build a network. Other than that, a
layer in Leaf can implement any behaviour.

Layers are constructed via the [`LayerConfig`
(/src/layer.rs)][layer-config], which makes creating even complex networks easy
and manageable.

```rust
// construct the config for a fully connected layer with 500 notes
let linear_1: LayerConfig = LayerConfig::new("linear1", LinearConfig { output_size: 500 })
```

A `LayerConfig` can be turned into an initialized, fully operable [`Layer`
(/src/layer.rs)][layer] with its `from_config` method.

```rust
// construct the config for a fully connected layer with 500 notes
let linear_1: LayerConfig = LayerConfig::new("linear1", LinearConfig { output_size: 500 })
let linear_network_with_one_layer: Layer = Layer::from_config(backend, &linear_1);
```

Hurray! We just constructed a [network](./deep-learning-glossary.html#Network)
with one layer. (In the following chapter we will learn how to create more
powerful networks). 

The `from_config` method initializes a `Layer`, which wraps the specific implementation (a struct that has  [`ILayer`(/src/layer.rs)][ilayer] implemented) in a worker field.
In the tiny example above, the worker field of the `linear_network_with_one_layer`
is a [`Linear` (/src/layers/common/linear.rs)][linear-layer] because we constructed
the `linear_network_with_one_layer` from a `LinearConfig`. The worker field
introduces the specific behaviour of the layer.

In the following chapters we explore more about how we can construct
real-world networks, the layer lifecycle and how we can add new layers to the Leaf framework.

[layer-config]: https://github.com/autumnai/leaf/blob/master/src/layer.rs
[layer]: https://github.com/autumnai/leaf/blob/master/src/layer.rs
[ilayer]: https://github.com/autumnai/leaf/blob/master/src/layer.rs
[linear-layer]: https://github.com/autumnai/leaf/blob/master/src/layers/common/linear.rs

### What can Layers do?

A layer can implement basically any behaviour: deep learning related like
convolutions or LSTM, classical machine learning related like nearest neighbors
or random forest, or utility related like logging or normalization. To make the
behaviour of a layer more explicit, Leaf groups layers into one of five
categories based on their (machine learning) functionality:

1) [Activation](#Activation&#32;Layers)
2) [Common](#Common&#32;Layers)
3) [Loss](#Loss&#32;Layers)
4) [Utility](#Utility&#32;Layers)
5) [Container.](#Container&#32;Layers)

In practice, the groups are not really relevant, it helps make the file
structure cleaner. And it simplifies the explanation of what a layer is
doing.

#### Activation Layers

Activation layers provide element-wise operations and return an output of
the same size as the input. Activation layers can be seen as equivalent to
nonlinear [Activation Functions](https://en.wikipedia.org/wiki/Activation_function)
and are a fundamental piece in neural networks.

Examples of activation layers are `Sigmoid`, `TanH` or `ReLU`. All available
activation layers can be found at
[src/layers/activation](https://github.com/autumnai/leaf/tree/master/src/layers/activation).

#### Loss Layers

Loss layers compare an output to a target value and assign a cost to minimize.
Loss layers are often the last layer in a network.

Examples of loss layers are `Hinge Loss`, `Softmax Loss` or `Negative Log
Likelihood`. All available loss layers can be found at
[src/layers/loss](https://github.com/autumnai/leaf/tree/master/src/layers/loss).

#### Common Layers

Common layers can differ in their connectivity and behavior. They are typically
anything that is not an activation or loss layer.

Examples of common layers are `fully-connected`, `convolutional`, `pooling`, `LSTM`,
etc. All available common layers can be found at
[src/layers/common](https://github.com/autumnai/leaf/tree/master/src/layers/common).

#### Utility Layers

Utility layers introduce all kind of helpful functionality, which might not be
directly related to machine learning and neural nets. These could be operations
for normalizing, restructuring or transforming information, log and debug
behavior or data access. Utility Layers follow the general behavior of a layer
like the other types.

Examples of Utility layers are `Reshape`, `Flatten` or `Normalization`. All
available utility layers can be found at
[src/layers/utility](https://github.com/autumnai/leaf/tree/master/src/layers/utility).

#### Container Layers

Container layers take `LayerConfig`s and connect them on initialization, which
creates a "network". But as container layers are layers themselves, one can stack multiple
container layers on top of another and compose even bigger container layers.
Container layers differ in how they connect the layers that it receives.

Examples of container layers are `Sequential`. All available container layers
can be found at
[src/layers/container](https://github.com/autumnai/leaf/tree/master/src/layers/container).

### Why Layers?

The benefit of using a layer-based design approach is that it allows for a very expressive
setup that can represent, as far as we know, any machine learning algorithm.
That makes Leaf a framework, that can be used to construct practical machine
learning applications that combine different paradigms.

Other machine learning frameworks take a symbolic instead of a layered approach.
For Leaf we decided against it, as we found it easier for developers to work with
layers than mathematical expressions. More complex algorithms like LSTMs are
also harder to replicate in a symbolic framework. We
believe that Leafs layer approach strikes a great balance between
expressiveness, usability and performance.
