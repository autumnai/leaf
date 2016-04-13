# Glossary

### Layer

**In General**

A layer is the highest-level building block in a (Deep) Neural Network. A layer
is a container that usually receives weighted input, transforms it and returns
the result as output to the next layer. A layer usually contains one type of
function like ReLU, pooling, convolution etc. so that it can be easily compared
to other parts of the network. The first and last layers in a network are called
input and output layers, respectively, and all layers in between are called
hidden layers.

**In Leaf**

In Leaf, a layer is very similar to the general understanding of a layer. A layer
in Leaf, like a layer in a (Deep) Neural Network,

* is the highest-level building block
* needs to receive input, might transform it and needs to return the result
* should be uniform (it does one type of function)

Additionally to a Neural Network layer, a Leaf layer can implement any
functionality, not only those related to Neural Networks like ReLU, pooling,
LSTM, etc. For example, the `Sequential` layer in Leaf, allows it to connect
multiple layers, creating a network.

### Network

**In General**

A network, also often called Neural Network (NN) or Artificial Neural Network
(ANN) is a subset of Machine Learning methods.

A not exhaustive list of other Machine Learning methods:  
*Linear Regression, SVM, Genetic/Evolution Algorithms, dynamic programming,
deterministic algorithmic optimization methods.*

**In Leaf**

In Leaf, a network means a graph (a connected set) of one or more
[layers](./layers.html). This network can consist of Artificial Neural Network
methods, other Machine Learning methods or any other (not Machine Learning
related) methods. As described in [2. Layers](./layers.html) a network in Leaf
is actually a layer which connects other layers.

An initialized network is a network, which is ready to be executed, meaning it
is fully constructed e.g. all necessary memory is allocated on the host or device.
