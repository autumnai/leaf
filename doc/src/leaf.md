# Leaf

This short book will teach you about [Leaf][leaf], the Machine Intelligence Framework
engineered by software developers, not scientists. It was inspired by the
brilliant people behind TensorFlow, Torch,
Caffe, Rust and numerous research papers and brings modularity, performance and
portability to Deep Learning. Leaf has a very simple API, [Layers][layers] and
[Solvers][solvers], and is one of the fastest Machine Intelligence Frameworks
available.

> **Assumption**  
> The Leaf Book requires a basic understanding of the fundamental concepts
> of Machine and Deep Learning. Recommended resources are
>
> * [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
> * [Stanford Course on (Convolutional) Neural Networks](http://cs231n.github.io/)
> * [A 'brief' history of Deep Learning](http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/)
> * [The Glossary](./glossary.html)

Deep Learning is really easy. Construct a network by chaining layers and then train the
network by feeding it examples. That is why Leaf's entire API
consists of only two concepts: [Layers][layers] and [Solvers][solvers]. Layers to
construct almost any kind of network. Deep Networks and even classical, stochastic based
algorithms/networks. And Solvers for training and execution of the network.
That is already the entire API for Machine Learning with Leaf.

[leaf]: https://github.com/autumnai/leaf
[layers]: ./layers.html
[solvers]: ./solvers.html

## API Documentation

Alongside this book you can also read the [Rust API documentation][api-docs] if
you would like to use Leaf as a crate or write a new library on top of it and
need a more low-level overview.

[api-docs]: http://autumnai.github.io/leaf/

## License

Leaf is licensed under either of

* [Apache License v2.0](https://github.com/autumnai/leaf/blob/master/LICENSE-APACHE) or,
* [MIT license](https://github.com/autumnai/leaf/blob/master/LICENSE-MIT)

at your option.



