<div align="center">
  <img src="http://autumnai.com/images/autumn_leaf_benchmarks_alexnet.png"><br><br>
</div>
> Forward and Backward duration for Leaf 0.2 and other Machine Learning Frameworks on the popular Alexnet.

# Announcing Leaf 0.2

We are happy to announce today the release of Leaf 0.2 on which we have been
working on for the last weeks. Leaf is a modular, performant, portable
Machine Intelligence Framework.
It is the Hacker's Machine Intelligence Framework, developed by software
engineers.

You can [install Leaf 0.2][install] and [run examples][examples], including
popular Deep Neural Networks like Alexnet, Overfeat, VGG and more.

## What's in Leaf 0.2

The release was mostly about finding an efficient and clean architecture,
catching up with the performance level of other Machine Learning Frameworks. It
shares concepts from the brilliant work done by the people behind Torch,
Tensorflow, Caffe, Rust and numerous research papers. We have several large
features under development, Leaf 0.2 gives us the platform to go on exploring
new territory with Leaf 0.3.

### Performance

Leaf 0.2 is one of the fastest Machine Intelligence Frameworks that exist
today. Rust was a big help in developing the entire platform over the course of
a few months. We achieved a very efficient GPU utilization and oriented our
architecture close to Torch and achieved the distribution capabilities of
Tensorflow, on a lower abstraction level. More information in the
following sections. 

More Benchmarks and comparisons, including Memory utilization, can be found on
[Deep Learning Benchmarks][deep-learning-benchmarks-website].

### Portability

Leaf 0.2 uses [Collenchyma][collenchyma] for training and running models on
CPUs, GPUs, FPGAs, etc. with OpenCL or CUDA or other Computation Languages, on
various machines and operating systems, without the need to adapt your code what
so ever. This makes deployment of models to servers, desktops, smartphones and
later embedded devices very convenient.

With that abstraction and separation of algorithm representation and execution,
we gain a nice Framework for distributed model execution, without relying
on a symbolic, data-flow graph model like Tensorflow, which introduces
performance and development overhead concerns.

### Architecture

Leaf 0.2 replaces special `Network` objects with container layers
like the `Sequential` layer. Where previously all weights were stored centrally
by the Network, each Layer is now responsible for managing its own weights.
This allows for more flexibility in expressing different network architectures.
It also enables better programmatic generation of networks by nesting container
layers where each container represents a common pattern in neural networks,
e.g. Convolution, Pooling and ReLU following each other.

### Contributors for Leaf 0.2

We had 9 individual contributors, which made Leaf 0.2 possible. Thank you so
much for your contribution, when Leaf wasn't even executable, yet. And thank you
for everyone who took the time to engage with us on [Gitter][gitter-leaf] and
Github.

* Maximilian Goisser ([@hobofan](https://twitter.com/hobofan))
* Michael Hirn ([@mjhirn](https://twitter.com/mjhirn))
* Ewan Higgs ([ehiggs](https://github.com/ehiggs))
* Florian Gilcher ([@argorak](https://twitter.com/Argorak))
* Paul Dib ([pdib](https://github.com/pdib))
* David Irvine ([dirvine](https://github.com/dirvine))
* Pascal Hertleif ([killercup](https://github.com/killercup))
* Kyle Schmit ([kschmit90](https://github.com/kschmit90))
* Sébastien Lerique ([wehlutyk](https://github.com/wehlutyk))

<div align="center">
  <p>
    <a href="http://autumnai.com">More about Leaf and Autumn</a> |
    Follow on Twitter: <a href="https://twitter.com/autumn_eng">@autumn_eng</a>
  </p>
</div>

[install]: https://github.com/autumnai/leaf#getting-started
[examples]: https://github.com/autumnai/leaf#examples
[collenchyma]: https://github.com/autumnai/collenchyma
[deep-learning-benchmarks-website]: http://autumnai.com/deep-learning-benchmarks
[gitter-leaf]: https://gitter.im/autumnai/leaf
